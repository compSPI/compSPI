# Important files
* `ray_pipeline.py` is the core of the code: VAEGAN
* `Singularity` (devops_pipeline) is the recipe that tells which pipeline needs to be run : ray_pipeline.py
* `build_run_pipeline`: build the Singularity image and runs it
* `ray_slac_run_pipeline`: do it on slac's cluster.

# Installation

## Build Singularity image

### On Mac

Following [these steps](https://sylabs.io/guides/3.0/user-guide/installation.html#mac), you first need to install Vagrant using Brew. Then:
```
cd ~
mkdir vm-singularity
cd vm-singularity
vagrant init singularityware/singularity-2.4
vagrant up
vagrant ssh
vagrant@vagrant:~$ singularity --version
```
At that step it should print the installed version of Singularity: `2.4`. We can thus proceed to build our Singularity images.
```
vagrant destroy
cp Vagrantfile <path-to-compSPI-parent-directory>/
cd <path-to-compSPI-parent-directory>
```
Increase the allocated RAM (from 1GB to 4GB in this case) in `Vagrantfile`:
```
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "4096"
  end
```
Finally:
```
vagrant up
vagrant ssh
vagrant@vagrant:~$ cd ../../
vagrant@vagrant:~$ cd vagrant/compSPI/src
vagrant@vagrant:~$ ./build_run_pipeline.sh
```
