cd [root]
build_gym:
	python3 -m pip install -e .

tfboard:
	tensorboard --logdir logs/ppo_laser_tracker_tensorboard/
	# http://localhost:6006/


