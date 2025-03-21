import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load sound file (must be a .wav file)
pygame.mixer.music.load("file_example_WAV_1MG.wav")

# Play sound
print("Playing sound...")
pygame.mixer.music.play()

# Wait for the sound to finish playing
while pygame.mixer.music.get_busy():
    time.sleep(1)

print("Sound finished.")
