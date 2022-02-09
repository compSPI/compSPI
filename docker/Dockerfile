FROM docker.io/ubuntu:18.04

MAINTAINER Frederic Poitevin "fpoitevi@stanford.edu"

WORKDIR /work

# Install useful packages
RUN apt-get -y update && \
    apt-get -y install git vim emacs tmux screen unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Download miniconda installer, perform the installation and remove it
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /work/Miniconda3-latest-Linux-x86_64.sh && \
    /work/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda && \
    rm /work/Miniconda3-latest-Linux-x86_64.sh

# Copy the code and generate the environment
# Remains to be done: COPY all and remove the fat later, instead-of multiplying layers:
COPY environment.yml /work/environment.yml
COPY README.md /work/README.md
COPY setup.py /work/setup.py
COPY pytest.ini /work/pytest.ini
COPY .codecov.yml /work/.codecov.yml
COPY .coveragerc /work/.coveragerc
COPY .flake8 /work/.flake8
COPY compSPI /work/compSPI
COPY tests /work/tests
RUN . /opt/anaconda/etc/profile.d/conda.sh && conda update -c conda-forge conda
RUN . /opt/anaconda/etc/profile.d/conda.sh && \
    conda env update --name base --file environment.yml && \
    conda activate base && \
    pip install -e . && \
    conda clean -afy

# Copy and add the entrypoint script
ADD docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]