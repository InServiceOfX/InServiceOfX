Creating torrents requires having `transmission-create` installed.

https://help.ubuntu.com/community/TransmissionHowTo

```
sudo apt-get install transmission-cli
```

## Seed a torrent

```
transmission-cli -w /path/to/files_or_directory my_torrent_file.torrent
```
