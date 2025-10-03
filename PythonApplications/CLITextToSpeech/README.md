## Use `ffmpeg` to create audio snippets of "source" audio files

### Create an audio snippet from start timestamp to end

```
ffmpeg -i 'drinks White Claw once - Trevor Wallace.mp3' -ss 01:00 -to 2:53 FakeTrevorWallace-en.wav

```
(Use `-t` if you want the audio snippet by specifying the time duration)

### Concatenate a few audio clips together

```
ffmpeg -i FakeAriGold-Season1-en-00.wav -i FakeAriGold-Season2-en-00.wav -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" FakeAriGold-en.wav
```