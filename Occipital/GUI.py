from tkinter import *
import Detection
from PIL import Image, ImageTk
def LPcalling(input1):
    ld = Detection()
    global label3
    result = ld.result(input1)
    print(result)
    label3 = Label(root, text='The language is: '.join(result), font=('Kanit', 14), bg='#2A282A', fg='#9C9C9C')
    label3.place(x=750, y=640)

root = Tk()
root.title('Language Predictor')
root.iconbitmap('pp.ico')
root.geometry('1107x787')

# Image Insertion
bg = Image.open('bg.png')
bg = ImageTk.PhotoImage(bg)
label2 = Label(root, image = bg)
label2.pack()

# Inserting Label 1
label1 = Label(root, text = 'Having trouble recognizing the language?',
               font=('Kanit', 18), fg='#787278', bg='#2A282A')
label1.place(x=565, y=345)

# Inserting Label 2
label2 = Label(root, text='Enter Here:',
               font=('Kanit', 12), bg='#2A282A', fg='#787278')
label2.place(x=572, y=470)

# Taking input string
input1 = Entry(root, width=40, font=('Kanit', 14), borderwidth=0, bg='#424242', fg='#787278')
input1.place(x=572, y=500)

# Creating Button
nb_button = Button(root, text="Predict", fg='#9C9C9C', bg='#424242', borderwidth=0, font=('Kanit', 14),
                   command=lambda: LPcalling(input1.get()), padx=10)
nb_button.place(x=740, y=550)

# Inserting Label 3
label3 = Label(root, text='The language is:',
               font=('Kanit', 8), bg='#424242', fg='#9C9C9C')

root.mainloop()




