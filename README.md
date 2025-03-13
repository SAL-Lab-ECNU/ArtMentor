<div align="center">

<div align="center">
    <img src="./artmentor_logo.png" width="60px"/>
</div>

**ArtMentor: AI-Assisted Evaluation of Artworks to Explore Multimodal Large Language Models Capabilities**

</div>

## Overview

This repository contains the code for **ArtMentor**, a tool designed to assist art teachers in evaluating elementary student artworks using AI. ArtMentor leverages the capabilities of large multimodal language models (MLLMs) to provide insights into creative processes and educational value.

- **Paper**: [ArtMentor: AI-Assisted Evaluation of Artworks to Explore Multimodal Large Language Models Capabilities](./ArtMentor%20AI-Assisted%20Evaluation%20of%20Artworks%20to%20Explore%20Multimodal%20Large%20Language%20Models%20Capabilities.pdf) (CHI 2025)
- **Project Website** [ArtMentor](https://artmentor.github.io/)
- **Video Presentation** [Video](https://www.bilibili.com/video/BV1zkQpYPEAW/?share_source=copy_web&vd_source=7d5d8be206e3c589172a6b8dcccfbbcc)


---

## Contents

- [Overview](#overview)
- [Contents](#contents)
- [Run ArtMentor](#run-artmentor)
- [Advanced Usage](#advanced-usage)
    - [Explanation of Parameters:](#explanation-of-parameters)
- [Data Collection](#data-collection)

---

## Run ArtMentor

ArtMentor is a Flask app that serves requests from users, manages sessions, and stores logs for future analysis.

**1. Clone this repository**

To clone this repository, use the following command:

```
git clone https://github.com/ArtMentor/ArtMentorApp.git

```

Install the required packages:

```
pip install -r requirements.txt
```

**2. Add your API key(s)**

Create a file `./key.txt` and add your API key(s):

sk-***************************************

Replace the `sk-***************************************` with your OpenAI API key. If you don't have it, you can get one [here](https://openai.com/pricing).

**3. Run ArtMentor**

To start the server, simply open the `ArtMentor_app.py` file in PyCharm and click the green "Run" button (the triangle icon). PyCharm will handle the execution, and you'll see the application logs displayed in the console, indicating that the Flask server is up and running. The output will include details such as the Python executable being used, the application name (`ArtMentor_app`), and the server URL.

Once the server is running, you can access the application by navigating to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

Note: The specific console output may vary based on your environment settings.

## Advanced Usage

*1. Modifying API Keys and Proxy Settings*

The application uses OpenAI's API for generating responses. If you need to change the API key or the proxy settings:

- **API Key**: The API key is read from the `key.txt` file. To update the key, simply modify this file.
- **Proxy**: The proxy settings are configured directly in the code using `os.environ`. Update the `proxy` variable in the code if you need to change the proxy server.

```python
# Set OpenAI's API key and proxy
proxy = "http://127.0.0.1:7890"
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy

# Reading the API key from a text file
with open('key.txt', 'r') as file:
    api_key = file.read().strip()
client = openai.OpenAI(api_key=api_key)
```

*2. Customizing the Entity Recognition Agent*

The `Entity_Recognition_Agent` function processes the input image and generates descriptive labels. If you want to modify the behavior of this agent, such as changing the prompt structure or adjusting the model parameters, or parametric like `temperature`, `top_p`, and others. you can do so directly in this function.

```python
def Entity_Recognition_Agent(image_data):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": "Identify and list the objects or features present in the image using descriptive labels. ..."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{image_data}",
                         "detail": "high"
                     },
                     },
                ],
            }
        ],
        max_tokens=100,
    )
    # Process the response...
```

#### Explanation of Parameters:

- **`model`**: Specifies the language model to use.

  - **Possible Values**: `"gpt-4o"`, or other available models.
  - **Explanation**: Different models have varying levels of complexity, accuracy, and response times. Choose the model based on the task's requirements.
- **`temperature`**: Controls the randomness of the output.

  - **Value**: `0.0`.
  - **Explanation**:
    - A value of `0.0` makes the output highly deterministic, with the model choosing the most probable next word, which is suitable for tasks requiring precision and consistency.
    - Higher values (e.g., `1.0` or above) introduce more randomness and diversity into the output, useful for creative or open-ended tasks.
- **`top_p`**: Implements nucleus sampling by considering tokens with a cumulative probability up to this threshold.

  - **Range**: `0.0` to `1.0`.
  - **Explanation**:
    - With `top_p=1.0`, the model considers all possible tokens, generating more diverse outputs.
    - Lower values (e.g., `0.9`) restrict the model to focus only on the most probable tokens, making the output more focused and contextually relevant.
- **`max_tokens`**: Limits the maximum number of tokens in the generated response.

  - **Range**: Any positive integer, typically up to `4096` tokens depending on the model.
  - **Explanation**: This parameter controls the length of the response. For example, a value of `100` limits the response to 100 tokens, which is ideal for concise outputs. Larger values allow for more detailed responses.

*If there's anything you're still not sure about, click [here](https://beta.openai.com/docs/api-reference/completions/create) for more details.*

By adjusting these parameters, you can fine-tune how the model interacts with the input image, balancing between creativity, precision, and response length to best fit your application's needs.

*3. Adjusting the Review and Suggestion Prompts*

The system generates prompts for reviews and suggestions based on the provided labels, scores, and suggestions. These prompts are defined in the `create_Review_prompt` and `create_Suggestion_prompt` functions. You can modify these prompts to customize how the AI evaluates and suggests improvements for the artworks.

```python
def create_Review_prompt(labels_data, score_Review_data, dimension):
    # Modify the prompt construction logic here
    full_prompt = f"The suggestion dimension is {dimension}. You may receive some Reviews, score or suggestions feedback..."
    return full_prompt
```

*4. Handling File Uploads and Extensions*

The application supports specific file types for upload (e.g., PNG, JPG). If you need to support additional file formats or modify how files are handled, adjust the `allowed_file` function and related configuration.

```python
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
```

## Data Collection

**Evaluation Metrics**

The ArtMentor system uses several key metrics to assess the performance of multimodal large language models:

1. **Entity Classification Metrics**: Measures how accurately the model identifies visual elements in artworks.
2. **Score Acceptance Metrics**: Evaluates the consistency between AI-generated scores and human-modified scores.
3. **Text Acceptance Metrics**: Tracks the extent of user changes to AI-generated comments and suggestions.
4. **Art Style Metrics**: Assesses the model's ability to recognize and respond to different artistic styles.

**Interaction Data**

ArtMentor meticulously records all interactions between art educators and the system, capturing data on:

- **Entity Recognition**: A record of AI and user edits to tags, includes AI-generated tags, user-added tags, user-deleted tags, and a special ‘Style’ type tag's generation and deletion records.
- **Score Generation**: A record of AI and user edits to scores, including scores initially generated by the AI, scores generated by the AI in each round, and scores reviewed and edited by the user.
- **Review and Suggestion Generation**: A record of AI and user edits to Reviews and Suggestions, including Reviews and Suggestions generated by the AI in each round, Reviews and Suggestions reviewed and edited by the user, words deleted by the user, and words added by the user.

This interaction data is invaluable for understanding and improving the system's effectiveness in educational settings.

click [here](https://github.com/SAL-Lab-ECNU/ArtMentor/tree/main/ArtMentorAnalysis) for the experimental data we collected, and all the analysis codes.

**Artwork Source and Audio Descriptions**

For the initial analysis of the **ArtMentor** application, we used a collection of 20 artworks created by elementary school students from the same primary school. Along with the artworks, we also collected audio recordings where the students described their creative processes and thoughts behind each artwork.

When participants evaluated the artworks using the ArtMentor system, they were provided with these 20 artworks and the corresponding audio recordings, which were played during the evaluation to give participants additional context for their assessments.

Project Website is [here](https://artmentor.github.io/).

Our video presentation is [here](https://www.bilibili.com/video/BV1zkQpYPEAW/?share_source=copy_web&vd_source=7d5d8be206e3c589172a6b8dcccfbbcc).

Cite

```bash
@article{zheng2025artmentor,
  title={ArtMentor: AI-Assisted Evaluation of Artworks to Explore Multimodal Large Language Models Capabilities},
  author={Zheng, Chanjin and Yu, Zengyi and Jiang, Yilin and Zhang, Mingzi and Lu, Xunuo and Jin, Jing and Gao, Liteng},
  journal={arXiv preprint arXiv:2502.13832},
  year={2025}
}
```
