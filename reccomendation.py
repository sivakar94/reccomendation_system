import streamlit as st
import streamlit.components.v1 as stc

#Load EDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

def load_data(data):
    df= pd.read_csv(data)
    return df

def vectorize_text_to_cosine_mat(data):
    
    count_vect=CountVectorizer()
    cv_mat=count_vect.fit_transform(data)
    # get the cosine
    cosine_sim= cosine_similarity(cv_mat)
    return cosine_sim

#Reccomendation sys
def get_reccomendation(title, cosine_sim_mat, df, num_of_rec=5):
    # indices of the course
    course_indices= pd.Series(df.index, index=df['course_title']).drop_duplicates()
    # Index of course
    idx= course_indices[title]

    # Look into the cosine matrix for that index
    sim_scores= list(enumerate(cosine_sim_mat[idx]))
    sim_scores= sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices= [i[0] for i in sim_scores[1:]]
    selected_course_scores= [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df= df.iloc[selected_course_indices]
    result_df['similiarity_score']= selected_course_scores
    final_recommended_courses = result_df[['course_title','similarity_score','url','price','num_subscribers']]
	return final_recommended_courses.head()



def main():
    
    st.title("Course Recommendation APP")
    
    menu=["Home","Reccomend","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data("udemy_courses.csv")

    if choice == 'Home':
        st.subheader("Home")
        st.dataframe(df.head(10))
    
    elif choice == 'Reccomend':
        st.subheader("Reccomend Courses")
        
        search_term= st.text_input("Search")
        num_of_rec= st.sidebar.number_input("Number",4,30,7)
        cosine_sim_mat= vectorize_text_to_cosine_mat(df['course_title'])
        
        if st.button("Reccomend"):
            if search_term is not None:
                try:
                    result= get_reccomendation(search_term,cosine_sim_mat, df, num_of_rec)
                except:
                    result="Not found"
                
                st.write(result)
                for row in result.iterrows():
                    rec_title = row[0][0]
                    rec_score = row[0][1]
                    rec_url= row[0][2]
                    rec_price= row[0][3]
                    rec_num_sub= row[0][4]

                    st.write("Title",rec_title,)





    else:
        st.subheader("About")
        st.text("Built With Stramlit and Pandas")

if __name__=='__main__':
    main()


### 