rm -rf actor
mkdir actor
mkdir actor/deletedexp
mkdir actor/episodescore
mkdir model
mkdir frames
mkdir learner
mkdir learner/sampleexp
mkdir learner/replaymemory

sudo apt-get install python3-pip
pip3 install numpy
pip3 install moviepy
pip3 install matplot
pip3 install scipy
git clone https://github.com/openai/gym.git
cd gym
pip3 install gym
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
pip3 install -e '.[atari]'
