pip install Cython pillow lxml jupyter matplotlib
pip install pycocotools

# Protobuf Compilation
cd research
protoc object_detection/protos/*.proto --python_out=.
cd ..