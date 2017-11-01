from argparse import ArgumentParser
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit, stdout

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         save_xarray, make_tetrode_dataframe)
from src.analysis import decode_ripple_clusterless, detect_epoch_ripples
from src.parameters import ANIMALS, PROCESSED_DATA_DIR, SAMPLING_FREQUENCY


def decode_ripples(epoch_key):
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, sampling_frequency=SAMPLING_FREQUENCY)

    # Compare different types of ripples
    replay_info, state_probability, posterior_density = (
        decode_ripple_clusterless(epoch_key, ANIMALS, ripple_times,
                                  mark_names=None))

    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)

    results = dict()
    results['replay_info'] = replay_info.reset_index().to_xarray()
    results['position_info'] = position_info.to_xarray()
    results['state_probability'] = state_probability
    results['posterior_density'] = posterior_density

    for group_name, data in results.items():
        save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)


def decode_replay_by_brain_area(epoch_key):
    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)

    results = dict()

    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)
    results['position_info'] = position_info.to_xarray()

    for brain_area in tetrode_info.area.dropna().unique().tolist():
        if brain_area not in ['???', 'Reference']:
            print(brain_area)
            try:
                ripple_times = detect_epoch_ripples(
                    epoch_key, ANIMALS, sampling_frequency=SAMPLING_FREQUENCY,
                    brain_areas=brain_area)
                # Compare different types of ripples
                replay_info, state_probability, posterior_density = (
                    decode_ripple_clusterless(
                        epoch_key, ANIMALS, ripple_times, mark_names=None,
                        brain_areas=brain_area))

                results[brain_area + '/replay_info'] = (
                    replay_info.reset_index().to_xarray())
                results[brain_area + '/state_probability'] = state_probability
                results[brain_area + '/posterior_density'] = posterior_density
            except (ValueError, FileNotFoundError):
                continue

    for group_name, data in results.items():
        try:
            save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)
        except KeyError:
            continue


def decode_replay_during_hippocampus_ripple(epoch_key):
    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, sampling_frequency=SAMPLING_FREQUENCY,
        brain_areas='CA1')

    for brain_area in tetrode_info.area.dropna().unique().tolist():
        if brain_area not in ['???', 'Reference']:
            print(brain_area)
            try:
                # Compare different types of ripples
                replay_info, state_probability, posterior_density = (
                    decode_ripple_clusterless(
                        epoch_key, ANIMALS, ripple_times, mark_names=None,
                        brain_areas=brain_area))

                results = dict()
                name = 'hippocampal_ripple/' + brain_area
                results[name + '/replay_info'] = (replay_info.reset_index()
                                                  .to_xarray())
                results[name + '/state_probability'] = state_probability
                results[name + '/posterior_density'] = posterior_density

                for group_name, data in results.items():
                    save_xarray(
                        PROCESSED_DATA_DIR, epoch_key, data, group_name)
            except (ValueError, KeyError):
                continue


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=DEBUG,
        default=INFO,
    )
    return parser.parse_args()


def get_logger():
    formatter = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = StreamHandler(stream=stdout)
    handler.setFormatter(formatter)
    logger = getLogger()
    logger.addHandler(handler)
    return logger


def main():
    args = get_command_line_arguments()
    logger = get_logger()
    logger.setLevel(args.log_level)

    def _signal_handler(signal_code, frame):
        logger.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logger.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logger.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    decode_replay_by_brain_area(epoch_key)
    decode_replay_during_hippocampus_ripple(epoch_key)

    logger.info('Finished Processing')


if __name__ == '__main__':
    exit(main())
