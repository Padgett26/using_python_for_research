import string

alphabet = " " + string.ascii_letters + string.punctuation

positions = {}
for letter in alphabet:
    positions[letter] = alphabet.index(letter)


def encode(message, positions, shift):
    encoded = ""
    for letter in message:
        if letter in positions:
            index = positions[letter]
            if index + shift >= len(positions):
                index -= len(positions)
            elif index + shift < 0:
                index += len(positions)
            encoded += alphabet[index + shift]
    return encoded


message = "Hi, my name is caesar."
shift = 3

encoded = encode(message, positions, shift)
print(encoded)
decoded = encode(encoded, positions, -shift)
print(decoded)
