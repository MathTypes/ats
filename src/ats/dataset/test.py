import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset, port=8080, remote=True)

## create a session that doesn't display automatically
#session = fo.launch_app(dataset, auto=False)

print(session.config)

# https://voxel51.com/blog/fiftyone-tips-and-tricks-for-customizing-your-computer-vision-workflows-mar-03-2023/
# customize model
# https://github.com/voxel51/fiftyone/pull/2949
session.wait()
