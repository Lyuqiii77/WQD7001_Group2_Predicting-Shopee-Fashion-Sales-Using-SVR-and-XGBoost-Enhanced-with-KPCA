import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# Globals shared across functions
loc = 0
ax = None
canvas = None
tree = None
fig = None
df_global = None


def after_Table():
    global loc, ax, canvas, tree, df_global
    loc = (loc + 1) % 8
    ax.cla()

    if loc == 0:
        # Table of data types
        for i in tree.get_children():
            tree.delete(i)
        type_data = [type(df_global.loc[~df_global[col].isnull(), col].iloc[0]) for col in df_global.columns]
        for row in type_data:
            tree.insert("", tk.END, values=row)
    if loc == 1:
        # Heatmap of missing values
        sns.heatmap(df_global.isnull(), cbar=False, yticklabels=False, ax=ax)
        canvas.draw()
    if loc == 2:
        # Boxplot (example)
        cutted=df_global.loc[:,['price_ori', 'item_rating', 'price_actual', 'total_rating','total_sold', 'favorite', 'fees']]
        sns.heatmap(data=cutted.corr(),annot=True,cmap='viridis')
        canvas.draw()

    if loc == 3:
        sns.heatmap(df_global.isnull(), cbar=False, yticklabels=False, ax=ax)
        plt.title("missing data")
        canvas.draw()
    if loc == 4:
        grouped = df_global.groupby('Type of product').agg('sum').reset_index()
        sns.barplot(data=grouped, x='Type of product', y='total_sold', ax=ax)
        plt.title("volume sold")
        canvas.draw()
    if loc == 5:
        sns.boxplot(data=df_global[(df_global['price_actual']<100) ], x='item_rating', y='price_actual', ax=ax)
        plt.title("price_distribution")
        canvas.draw()
    if loc==6:
        sns.boxplot(data=df_global[(df_global['price_actual']>=100) & (df_global['price_actual']<1000) ], x='item_rating', y='price_actual', ax=ax)
        plt.title("price_distribution")
        canvas.draw()
    if loc==7:
        sns.boxplot(data=df_global[(df_global['price_actual']>=1000) ], x='item_rating', y='price_actual', ax=ax)
        plt.title("price_distribution")
        canvas.draw()
    


def EDA_pro_GUI(df):
    global ax, canvas, tree, fig, df_global

    df_global = df  # Store for use in callback

    app = tk.Tk()
    app.title("EDA")
    # Button
    btn = tk.Button(app, text='NEXT EDA', command=after_Table)
    btn.pack()
    #create a table of data_summary
    
    tree = ttk.Treeview(app, columns=list(df.columns), show='headings')
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    type_data = [type(df.loc[~df[col].isnull(), col].iloc[0]) for col in df.columns]
    for row in type_data:
        tree.insert("", tk.END, values=[row])
    tree.pack(expand=True, fill=tk.BOTH)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 5))
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    app.mainloop()
