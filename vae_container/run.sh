xhost +local:root

docker run -it --rm --net host --gpus all --shm-size=2gb \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTHORITY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -v ./Vae:/root/vae_environment \
    --name vae_container_live \
    vae_container_live bash
