cp ./update-nightly.sh /home/ubuntu
sudo crontab -l > ./crontab-temp
sudo echo "0 0 * * * /home/ubuntu/update-nightly.sh" >> ./crontab-temp
sudo crontab ./crontab-temp
