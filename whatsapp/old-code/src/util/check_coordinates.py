import pyautogui

cont = 'Y'
while cont == 'Y' or cont == 'y':

    # Wait for user input to determine the coordinates of a point
    input("Move the mouse cursor to the desired location and press Enter...")

    # Get the current mouse cursor position
    x, y = pyautogui.position()
    print("Coordinates:", x, y)
    cont = input("do you wanna continue? (Y/N)")



