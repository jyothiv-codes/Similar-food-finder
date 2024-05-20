import getpass, os, requests
import streamlit as st
import re
from google.cloud import aiplatform
import vertexai, json, requests
from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image
from astrapy.db import AstraDB, AstraDBCollection
from langchain.chat_models import ChatVertexAI
from langchain.schema.messages import HumanMessage
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)

def prediction():
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    # Initialize our vector db
    astra_db = AstraDB(token="YOUR-TOKEN-HERE")

    collection = AstraDBCollection(collection_name="food_match_v4", astra_db=astra_db)
    llm = ChatVertexAI(project="food-recommendations-vertex", model_name="gemini-pro-vision", region="uswest-1",toxicity_threshold=0.7)
    import json
    img = Image.load_from_file('food.png')
    embeddings = model.get_embeddings(image=img, contextual_text="solid food")

    # Perform the vector search against AstraDB Vector
    documents = collection.vector_find(
        embeddings.image_embedding,
        limit=3,
    )

    related_products_csv = "name, image\n"
    for doc in documents:
        related_products_csv += f"{doc['name']}, {doc['image']},\n"

    image_message = {
        "type": "image_url",
        "image_url": {"url": "food.png"},
    }
    text_message = {
        "type": "text",

        "text": f"What is this image? Find similar dishes with the highest match percentage and list the match percentage. The matched food's image should not be same as the current one. Use information only from the related info that is provided here"+related_products_csv+"if no relevant recommendations, then say, I couldnt find any. Do not repeat the sentences. If a food is found, fetch its URL as well.",
    }
    message = HumanMessage(content=[text_message, image_message])
    output = llm([message])
    url_pattern = r'https?://\S+'
    st.write(output.content)
    # Use re.findall to extract all URLs from the text
    urls = re.findall(url_pattern, str(output))
    

    print(urls[0])
    curr_url=urls[0]
    file_id=curr_url.split("&id=")[1][:-2]
    print("FILE ID---------",file_id)
   

    # URL
    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    response = requests.get(url)
    st.image(response.content)
    # URL
    



os.environ["GCP_PROJECT_ID"]="YOUR-PROJECT-ID"
os.environ["ASTRA_DB_ENDPOINT"] = "YOUR-DB-ENDPOINT"
os.environ["ASTRA_DB_TOKEN"] = "YOUR-TOKEN"
source_img_data = requests.get('your image').content
with open('food.png', 'wb') as handler:
  handler.write(source_img_data)

from langchain.chat_models import ChatVertexAI
from langchain.schema.messages import HumanMessage
import os, sys
st.title("Upload a food image here")

    # File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


submit_button = st.button("Submit")

    # Processing logic triggered by button click
if submit_button:
    # Process the input data

    if uploaded_file is not None:
          # Save the uploaded image to a file
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())

          # Display the selected image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        llm = ChatVertexAI(project=os.getenv("GCP_PROJECT_ID"), model_name="gemini-pro-vision", region="uswest-1",toxicity_threshold=0.7)

        image_message = {
            "type": "image_url",
            "image_url": {"url": "food.png"},
        }
        text_message = {
            "type": "text",
            "text": "What is this image?",
        }
        message = HumanMessage(content=[text_message, image_message])

        
        
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        prediction()


        


       