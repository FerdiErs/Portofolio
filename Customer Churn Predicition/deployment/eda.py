import streamlit as st 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image

st.set_page_config(
    page_title = 'Customer Churn Predictor'
)

def run():

    # Membuat Title 
    st.title('Customer Churn Predictor')

    #Sub header 
    st.subheader('Description for Customer Churn Predictor') 

    # Insert Gambar 
    image = Image.open('music.jpg')
    st.image(image, caption ='Dengar')

    #description 
    st.write('The goals of this churn estimator')
    st.write('Dengar is a music streaming platform that ask data scientist to predict will the customer churn')
    st.write('With this model we hope Dengar will be more focused with their goals')
    st.markdown('---')
    
    st.write('This page is created to show the visualization of the dataset')

    st.markdown('---')




    #show dataframe
    st.write('Dataset') 
    dup = pd.read_csv('https://raw.githubusercontent.com/FerdiErs/SQL/main/churn.csv')
    st.dataframe(dup)

    #visualization Function 

    def plot_hist(data, title, x_label):
        #create hist plot
        fig = plt.figure(figsize=(7, 5))
        sns.histplot(data, kde=True, bins=20, edgecolor='black')

        #Title and Labels
        st.title(title)

        st.pyplot(fig)

    def plot_countplot_with_numbers(data, x, hue, title, palette, figsize=(7, 5)):
        # Create CountPlot
        fig = plt.figure(figsize=figsize)
        g = sns.countplot(x=x, hue=hue, data=data, palette=palette)

        # Rotate x labels and move legend outside of the plot
        g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

        # Number in visualization
        for p in g.patches:
            height = p.get_height()
            g.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=10)

        # Title and labels
        st.title(title)

        st.pyplot(fig)

    #Age Distribution
    plot_hist(data=dup['age'], title='Age distribution', x_label='age')
    st.write('We can see that dengar had a distribution of age from 10-60')

    #Time Spent
    plot_hist(data=dup['avg_time_spent'], title='Time Spent', x_label='avg_time_spent')

    #pie chart customer region 
    st.write('### Customer Region Distribution')
    reg = dup.region_category.value_counts()
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct
    # Define a custom color palette
    colors = plt.cm.tab20c.colors
    fig = plt.figure(figsize=(5,5))
    reg.plot.pie(autopct=make_autopct(reg), startangle=90, colors=colors)
    plt.title('Customer Region')
    plt.axis('equal')
    st.pyplot(fig)
    st.write('We can see that dengar had 3 region with the most users from town')

     #Memberhsip based on Region
    plot_countplot_with_numbers(x='membership_category',hue='region_category', title='Memberhsip based on Region', data=dup, palette='flare', figsize=(7, 5))

    #membuat pie chart churn risk
    #count churn
    ch = dup.churn_risk_score.value_counts()

    # Define a custom color palette
    colors = plt.cm.Set3.colors

    # plot the data
    fig = plt.figure(figsize=(5,5))
    ch.plot.pie(autopct=make_autopct(ch), startangle=90, colors=colors)
    plt.title('Churn Risk')
    plt.axis('equal')
    st.pyplot(fig)
    st.write('We can see from the data that most users in Dengar will churn')
    

    #churn risk based on gender
    plot_countplot_with_numbers(data=dup, x='gender', hue='churn_risk_score', title='Churn Risk based on gender', palette='crest', figsize=(7, 5))
    
    #churn risk based on membership
    plot_countplot_with_numbers(data=dup, x='membership_category', hue='churn_risk_score', title='Churn Risk based on Membership', palette='flare', figsize=(7, 5))

   


if __name__== '__main__':
    run()