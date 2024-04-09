import customtkinter as ck
from tkinter import filedialog
import pandas as pd
import joblib
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

#function for selecting input file
def selectInputFile():
    inputFilePath = filedialog.askopenfilename()
    if inputFilePath:
        inputEntry.delete(0, ck.END)
        inputEntry.insert(ck.END, inputFilePath)

#function for selecting output destination
def selectOutputFile():
    outputFilePath = filedialog.asksaveasfilename(defaultextension=".csv")
    if outputFilePath:
        outputEntry.delete(0, ck.END)
        outputEntry.insert(ck.END, outputFilePath)

#function to generate output file with predictions
def generateOutput():
    inputFile = inputEntry.get()
    if inputFile:
        outputFile = outputEntry.get()
        if outputFile:
            #read csv as pandas dataframe
            df = pd.read_csv(inputFile)

            #load the saved models
            clf = joblib.load('logisticRegressionModel2.joblib')
            pca_pipe = joblib.load('pcaPipeline2.joblib')
            sc = joblib.load('standardScalar2.joblib')

            #perform standard scal
            dfTransformed = pca_pipe.transform(sc.transform(df))

            #generate predictions
            predictions = clf.predict(dfTransformed)

            #add predictions as a new column in the dataframe
            df['Predictions'] = predictions

            df.to_csv(outputFile, index=False)
            status = ck.CTkLabel(app, text="Success. File generated.", font=("Roboto", 18), text_color="green")
            status.pack(padx=5, pady=5)
            
            #plotting
            fig, ax = plt.subplots(1, figsize=(10, 10))
            FCTA = FigureCanvasTkAgg(fig, master=app)

            ax.clear()
            ax.plot(df.iloc[:, 0], df.iloc[:, 4], marker='o', linewidth=2, markersize=2)
            ax.set_title('Acceleration vs Time')
            ax.set_xlabel('Times(s)')
            ax.set_ylabel('Acceleration(m/s^2)')
            FCTA.draw()
            FCTA.get_tk_widget().pack()

            # Adding navigation toolbar
            toolbar = NavigationToolbar2Tk(FCTA, app)
            toolbar.update()
            FCTA.get_tk_widget().pack()

        else:
            status = ck.CTkLabel(app, text="Error. Invalid output path.", font=("Roboto", 18), text_color="red")
            status.pack(padx=5, pady=5)
    else:
        status = ck.CTkLabel(app, text="Error. Please select a CSV file.", font=("Roboto", 18), text_color="red")
        status.pack(padx=5, pady=5)

ck.set_appearance_mode("Light")
ck.set_default_color_theme("dark-blue")

app = ck.CTk()

#window height and width
app.geometry("600x800")

app.title("Jump / Walk Classifier")

#app title label
appTitle = ck.CTkLabel(app, text="Welcome to the Walk / Jump Classifier", font=("Roboto", 24))
appTitle.pack(padx=10, pady=10)

#input label
inputLabel = ck.CTkLabel(app, text="Select Input CSV File:", font=("Roboto", 18))
inputLabel.pack(padx=10, pady=10)

inputEntry = ck.CTkEntry(app, width=300)
inputEntry.pack(padx=5, pady=5)

#button to browse input files
inputButton = ck.CTkButton(app, text="Browse", command=selectInputFile)
inputButton.pack(padx=5, pady=5)

#output label
outputLabel = ck.CTkLabel(app, text="Select Output Destination:", font=("Roboto", 18))
outputLabel.pack(padx=10, pady=10)

outputEntry = ck.CTkEntry(app, width=300)
outputEntry.pack(padx=5, pady=5)

#button to browse output location
outputButton = ck.CTkButton(app, text="Browse", command=selectOutputFile)
outputButton.pack(padx=5, pady=5)

#button to generate output
button = ck.CTkButton(app, text="Generate Output File", font=("Roboto", 22), command=generateOutput)
button.pack(padx=10, pady=10)

app.mainloop()
