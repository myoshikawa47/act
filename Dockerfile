FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

# Install basic tools and Python 3.12 from Ubuntu 24.04.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    locales \
    python3 \
    python3-dev \
    python3-venv \
    software-properties-common \
    sudo \
    tmux \
    vim \
    wget \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Configure ROS One apt repository.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -sSL https://ros.packages.techfak.net/gpg.key \
        -o /etc/apt/keyrings/ros-one-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros1.list \
    && echo "# deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net $(lsb_release -cs) main-dbg" \
        >> /etc/apt/sources.list.d/ros1.list

# Install ROS One desktop and common ROS development tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-catkin-tools \
    python3-vcstool \
    ros-one-desktop \
    && rm -rf /var/lib/apt/lists/*

# Configure rosdep for ROS One.
RUN rosdep init || true \
    && echo "yaml https://ros.packages.techfak.net/ros-one.yaml one" \
        > /etc/ros/rosdep/sources.list.d/1-ros-one.list \
    && rosdep update

# Create a uv-managed virtual environment using the system Python 3.12.
# System site packages are enabled so ROS Python modules installed by apt remain visible.
RUN uv venv /opt/venv --python /usr/bin/python3.12 --system-site-packages

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Create workspace.
RUN mkdir -p /workspace/act
WORKDIR /workspace

# Create entrypoint.
RUN cat <<'EOF' > /ros_entrypoint.sh
#!/usr/bin/env bash
set -e

if [ -f /opt/ros/one/setup.bash ]; then
    source /opt/ros/one/setup.bash
fi

if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
fi

exec "$@"
EOF

RUN chmod +x /ros_entrypoint.sh

# Source ROS and uv environment automatically for interactive shells.
RUN cat <<'EOF' >> /root/.bashrc

if [ -f /opt/ros/one/setup.bash ]; then
    source /opt/ros/one/setup.bash
fi

if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
fi
EOF

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]