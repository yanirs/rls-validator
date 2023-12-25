# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = 2048
  end

  # Streamlit default port
  config.vm.network "forwarded_port", guest: 8501, host: 8501, host_ip: "127.0.0.1"

  config.vm.provision "shell", name: "apt dependencies", inline: <<-SHELL
    apt-get update
    apt-get install -y pipx
  SHELL

  config.vm.provision "shell", name: "poetry", privileged: false, inline: <<-SHELL
    pipx install poetry==$(cat /vagrant/.poetry-version)
  SHELL

  config.vm.provision "shell", name: "python dependencies", privileged: false, inline: <<-SHELL
    set -e
    cd /vagrant
    poetry install
    poetry run pre-commit install
    echo "Running Streamlit in screen ('screen -r streamlit'), check http://localhost:8501/"
    screen -dmS streamlit bash -c "poetry run streamlit run rlsv/app.py"
  SHELL

  config.vm.provision "shell", name: "apt-upgrade", run: "always", inline: <<-SHELL
    apt-get update && apt-get upgrade -y
  SHELL
end
