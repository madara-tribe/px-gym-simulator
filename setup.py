from setuptools import setup

setup(
    name='gym_laser_tracker',
    version='0.0.1',
    install_requires=['gym', 'numpy'],
    packages=['gym_laser_tracker', 'gym_laser_tracker.envs'],
)

