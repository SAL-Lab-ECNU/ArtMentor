import base64
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from matplotlib import rcParams
from werkzeug.utils import secure_filename
import os
import openai
from io import BytesIO
import re
import json
import difflib
# Set the global font
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12  # Set the font size
app = Flask(__name__)
app.secret_key = os.urandom(24)
print(f'Secret Key: {app.secret_key}')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# Set up OpenAI's API key and proxy
proxy = "http://127.0.0.1:7890"
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
# Read the API key from a text file
with open('key.txt', 'r') as file:
    api_key = file.read().strip()
client = openai.OpenAI(api_key=api_key)
# Global variables store rating data
global_scores = {
    'Realism': 0,
    'Deformation': 0,
    'Imagination': 0,
    'Color Richness': 0,
    'Color Contrast': 0,
    'Line Combination': 0,
    'Line Texture': 0,
    'Picture Organization': 0,
    'Transformation': 0
}
# Round counter for each file
round_counters = {}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def Entity_Recognition_Agent(image_data):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": "Identify and list the objects or features present in the image using descriptive labels. Use simple, clear terms like 'Face', 'Black hair', 'Thick lips', 'Big ears', etc. Ensure that each label is descriptive and that labels are separated by the symbol (';'). For example: Face; Black hair; Thick lips; Big ears;. Also, identify the art style of the image with a label starting with '## Style:'."
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
    response_content = response.choices[0].message.content
    # Separate style tags from other objects (entities)
    objects = response_content.split(";")
    style_label = None
    cleaned_objects = []
    for obj in objects:
        obj = obj.strip()
        if obj.startswith("## Style:"):
            style_label = obj[len("## Style:"):].strip()  # 去掉标志符和前导空格
        elif obj:
            cleaned_objects.append(obj)
    print("cleaned_objects",cleaned_objects)
    print("Style Label:", style_label)
    return cleaned_objects, style_label
def create_Suggestion_prompt(labels_data, score_Review_data, suggestion_data, dimension):
    print("####create_Suggestion_prompt####")
    print("labels_data",labels_data)
    print("score_Review_data",score_Review_data)
    print("suggestion_data",suggestion_data)
    print("dimension",dimension)
    if labels_data:
        # Extract 'original' and remove the elements in 'removed'
        updated_Entities = labels_data["original"]
        updated_Entities = [tag for tag in updated_Entities if tag not in labels_data["removed"]]
        # Determine the insertion point as the second-to-last position of the list. If there are fewer than 2 list items, insert it at the end of the list.
        insert_index = max(len(updated_Entities) - 1, 0)  # Make sure there is at least one element, otherwise insert at the beginning
        # Insert all 'added' elements at the calculated positions
        for added_item in reversed(labels_data["added"]):  # Use reversed to ensure the correct order of addition
            updated_Entities.insert(insert_index, added_item)
        Entities = "; ".join(updated_Entities)
        Entities_prompt = f"The image contains the following entities: {Entities}. "
    else:
        Entities_prompt = ""
    if score_Review_data == {
        "scores": {"original": 0, "current": 0, "initGPTscore": None},
        "Reviews": {"original": "", "current": "", "added": "", "removed": ""},
    }:
        score_Review_prompt = ""
    else:
        scores = f"The current score which the human user submitted for this dimension is {score_Review_data['scores']['current']}."
        Reviews = f"The current Review  which the human user submitted for this dimension is: {score_Review_data['Reviews']['current']}"
        score_Review_prompt = f"{scores} {Reviews} "
    if suggestion_data == {"suggestions": {"original": "", "current": "", "added": "", "removed": ""}}:
        suggestions_prompt = ""
    else:
        suggestions = f"The suggestion which the human user submitted for this dimension is {suggestion_data['suggestions']['current']}."
        suggestions_prompt = f"{suggestions}"
    print("dimension",dimension)
    full_prompt = f"The suggestion dimension is {dimension}. You may receive some Reviews, score or suggestions feedback from the human user. please consider it more.{Entities_prompt}{score_Review_prompt}{suggestions_prompt}Need improve dimension is {dimension}\n According to dimension, output a dictionary and don't use special symbols such as line breaks for ease of processing:"+"{\"suggestion\":suggestion for artworks to improve"+" }"
    # Debug statement: print the generated prompt and its length
    print(f"Generated prompt for {dimension}:")
    print(full_prompt)
    print(f"Prompt length (in characters): {len(full_prompt)}")
    return full_prompt
def create_Review_prompt(labels_data, score_Review_data, dimension):
    print("####create_prompt####")
    print("labels_data",labels_data)
    print("score_Review_data",score_Review_data)
    print("dimension",dimension)
    if labels_data:
        # Extract 'original' and remove the elements in 'removed'
        updated_Entities = labels_data["original"]
        updated_Entities = [tag for tag in updated_Entities if tag not in labels_data["removed"]]
        # Determine the insertion point as the second-to-last position of the list. If there are fewer than 2 list items, insert it at the end of the list.
        insert_index = max(len(updated_Entities) - 1, 0)  # Make sure there is at least one element, otherwise insert at the beginning
        # Insert all 'added' elements at the calculated positions
        for added_item in reversed(labels_data["added"]):  # Use reversed to ensure the correct order of addition
            updated_Entities.insert(insert_index, added_item)
        Entities = "; ".join(updated_Entities)
        Entities_prompt = f"The artworks from child contains the following entities: {Entities}. "
    else:
        Entities_prompt = ""
    if score_Review_data == {
            "scores": {"original": 0, "current": 0, "initGPTscore": None},
            "Reviews": {"original": "", "current": "", "added": "", "removed": ""},
        }:
        score_Review_prompt = ""
    else:
        scores = f"The current score which the human user submitted for this dimension is {score_Review_data['scores']['current']}."
        Reviews = f"The current Review  which the human user submitted for this dimension is: {score_Review_data['Reviews']['current']}."
        score_Review_prompt = f"{scores} {Reviews} "
    dimension_prompt = ""
    print("dimension",dimension)
    if dimension == "Realism":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Realism' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Realism. This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
            "\n5: The artwork exhibits exceptional detail and precision in depicting Realism features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
            "\n4: The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly Realism, minor discrepancies in perspective or detail might be noticeable."
            "\n3: The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
            "\n2: The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
            "\n1: The artwork shows minimal attention to Realism details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
        )
    elif dimension == "Deformation":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Deformation' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Deformation. This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
            "\n5: The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
            "\n4: The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
            "\n3: The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
            "\n2: The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
            "\n1: The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
        )
    elif dimension == "Imagination":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Imagination' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Imagination. This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
            "\n5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
            "\n4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
            "\n3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
            "\n2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
            "\n1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
        )
    elif dimension == "Color Richness":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Color Richness' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Color Richness. This criterion assesses the use and range of colors to create a visually engaging experience."
            "\n5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
            "\n4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
            "\n3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
            "\n2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
            "\n1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Color Contrast' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Color Contrast. This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
            "\n5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
            "\n4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
            "\n3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
            "\n2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
            "\n1: The artwork lacks effective color contrast, making the piece visually unengaging."
        )
    elif dimension == "Line Combination":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Line Combination' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Line Combination. This criterion assesses the integration and interaction of lines within the artwork."
            "\n5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
            "\n4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
            "\n3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
            "\n2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
            "\n1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
        )
    elif dimension == "Line Texture":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Line Texture' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Line Texture. This criterion evaluates the variety and execution of line textures within the artwork."
            "\n5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
            "\n4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
            "\n3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
            "\n2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
            "\n1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
        )
    elif dimension == "Picture Organization":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Picture Organization' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Picture Organization. This criterion evaluates the overall composition and spatial arrangement within the artwork."
            "\n5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
            "\n4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
            "\n3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
            "\n2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
            "\n1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
        )
    elif dimension == "Transformation":
        dimension_prompt = (
            "Assess the student's artwork based on the 'Transformation' criterion. Provide a score (1-5) and a detailed Review explaining the assessment."
            "You may receive some Reviews, score feedback from the human user. please consider it more."
            "\nCriterion: Transformation. This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
            "\n5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
            "\n4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
            "\n3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
            "\n2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
            "\n1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
        )
    full_prompt = f"{Entities_prompt}{score_Review_prompt}{dimension_prompt}\n output a dictionary:"+"{"f" dimension:{dimension},score: from 1 to 5,Review:Review for artworks"+" }"
    # Debug statement: print the generated prompt and its length
    print(f"Generated prompt for {dimension}:")
    print(full_prompt)
    print(f"Prompt length (in characters): {len(full_prompt)}")
    return full_prompt
def extract_score_Review(response_content):
    data = json.loads(response_content)
    # Extract the required fields
    dimension = data['dimension']
    score = data['score']
    Review = data['Review']
    return score, Review
def extract_suggestion(response_content):
    data = json.loads(response_content)
    # Extract the required fields
    suggestion = data['suggestion']
    return suggestion
def Review_Generation_Agent(image_data, labels_data, score_Review_data,dimension):
    prompt = create_Review_prompt(labels_data, score_Review_data, dimension)
    print("########Review_Generation_Agent#######")
    print(prompt)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {"role": "system", "content": prompt
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
        max_tokens=500,
    )
    response_content = response.choices[0].message.content
    print(response_content)
    score, Review= extract_score_Review(response_content)
    print("score",score)
    print("Review",Review)
    return score, Review
def Suggestion_Generation_Agent(image_data, labels_data, score_Review_data, suggestion_data,dimension):
    prompt = create_Suggestion_prompt(labels_data, score_Review_data, suggestion_data, dimension)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {"role": "system", "content": prompt
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
        max_tokens=500,
    )
    response_content = response.choices[0].message.content
    print(response_content)
    suggestion = extract_suggestion(response_content)
    print("suggestion",suggestion)
    return suggestion
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            base64_image = encode_image(filepath)

            try:
                # Reset global rating data
                for key in global_scores:
                    global_scores[key] = 0

                # Check if there is already a entity file
                labels_filepath = os.path.join('userActions/Entities', f"{filename}_labels.json")
                if os.path.exists(labels_filepath):
                    with open(labels_filepath, 'r') as f:
                        labels_data = json.load(f)
                    identified_objects = labels_data['original']
                    style_data = labels_data['style']
                else:
                    # Identify objects in images and generate labels
                    identified_objects, style_label = Entity_Recognition_Agent(base64_image)
                    # Initialize label and style data
                    labels_data = {"original": identified_objects, "added": [], "removed": []}
                    print("style_label", style_label)
                    style_data = {"original": [style_label], "added": [], "removed": []}
                    print("style_data",style_data)
                    # Save entities (including styles) to JSON files
                    data_to_save = {"original": labels_data["original"], "added": [], "removed": [], "style": style_data}
                    with open(labels_filepath, 'w') as f:
                        json.dump(data_to_save, f)

                    print(f"Successfully saved labels and style to file: {labels_filepath}")

                # Initialize the score to 0
                initial_scores = [0] * 9  # Assume there are 9 dimensions
                radar_chart = plot_radar_chart(initial_scores)

                return render_template('result.html',
                                       image_data=base64_image,
                                       identified_objects=identified_objects,
                                       style_label=style_data['original'],  # Transfer art style label
                                       radar_chart=radar_chart,
                                       image_name=filename  # Passing file name
                                       )

            except Exception as e:
                print(e)
                flash(f"An unexpected error occurred: {str(e)}")
                return redirect(request.url)

    return render_template('upload.html')
@app.route('/evaluate_dimension', methods=['POST'])
def evaluate_dimension():
    data = request.json
    image_data = data['image_data']
    dimension = data['dimension']
    image_name = data['image_name']
    # Read entity information
    labels_filepath = os.path.join('userActions/Entities', f"{image_name}_labels.json")
    if os.path.exists(labels_filepath):
        with open(labels_filepath, 'r') as f:
            labels_data = json.load(f)
    else:
        labels_data = None
    # Read ratings and reviews
    score_Review_filepath = os.path.join('userActions/score_Review', f"{image_name}_{dimension}_score_Review.json")
    if os.path.exists(score_Review_filepath):
        with open(score_Review_filepath, 'r') as f:
            score_Review_data = json.load(f)
    else:
        score_Review_data = {
            "scores": {"original": 0, "current": 0, "initGPTscore": None},
            "Reviews": {"original": "", "current": "", "added": "", "removed": ""},
        }
    # Make sure the 'initGPTscore' key exists and is initialized
    if 'initGPTscore' not in score_Review_data['scores']:
        score_Review_data['scores']['initGPTscore'] = None
    # Save the data of the current round
    save_round_data(image_name, dimension, 'score_Review', score_Review_data)
    score, Review = Review_Generation_Agent(image_data, labels_data, score_Review_data, dimension)
    # If initGPTscore is not set yet, it is set to the initial score
    if score_Review_data['scores']['initGPTscore'] is None:
        score_Review_data['scores']['initGPTscore'] = score
    # Update and overwrite original review and rating
    score_Review_data['scores']['original'] = score
    score_Review_data['scores']['current'] = score
    score_Review_data['Reviews']['original'] = Review
    score_Review_data['Reviews']['current'] = Review
    score_Review_data['Reviews']['added'] = ""
    score_Review_data['Reviews']['removed'] = ""
    # Overwrite new ratings and reviews
    with open(score_Review_filepath, 'w') as f:
        json.dump(score_Review_data, f)
    # Update global rating data
    if score is not None:
        global_scores[dimension] = score
    # Print the final returned data
    print(f"Evaluating {dimension}: score={score}, Review={Review}")
    return jsonify({
        "score": score,
        "Review": Review,
        "initGPTscore": score_Review_data['scores']['initGPTscore']
    })
@app.route('/generate_suggestion', methods=['POST'])
def generate_suggestion():
    data = request.json
    image_data = data['image_data']
    dimension = data['dimension']
    image_name = data['image_name']
    try:
        # Read entity information
        try:
            labels_filepath = os.path.join('userActions/Entities', f"{image_name}_labels.json")
            if os.path.exists(labels_filepath):
                with open(labels_filepath, 'r') as f:
                    labels_data = json.load(f)
            else:
                labels_data = None
        except Exception as e:
            print(f"Error loading labels data: {e}")
            labels_data = None

        # Read scores and reviews
        try:
            score_Review_filepath = os.path.join('userActions/score_Review',
                                                  f"{image_name}_{dimension}_score_Review.json")
            if os.path.exists(score_Review_filepath):
                with open(score_Review_filepath, 'r') as f:
                    score_Review_data = json.load(f)
            else:
                score_Review_data = None
        except Exception as e:
            print(f"Error loading score/Review data: {e}")
            score_Review_data = None
        # Read suggestion information
        try:
            suggestion_filepath = os.path.join('userActions/suggestion', f"{image_name}_{dimension}_suggestion.json")
            if os.path.exists(suggestion_filepath):
                with open(suggestion_filepath, 'r') as f:
                    suggestion_data = json.load(f)
            else:
                suggestion_data = {"suggestions": {"original": "", "current": "", "added": "", "removed": ""}}

            # Ensure suggestion_data structure is complete
            if 'suggestions' not in suggestion_data:
                suggestion_data = {"suggestions": {"original": "", "current": "", "added": "", "removed": ""}}
        except Exception as e:
            print(f"Error loading suggestion data: {e}")
            suggestion_data = {"suggestions": {"original": "", "current": "", "added": "", "removed": ""}}

        # Save the existing suggestion data to the everyround data file (first thing)
        try:
            save_round_data(image_name, dimension, 'suggestion', suggestion_data)
        except Exception as e:
            print(f"Error saving round data (before generation): {e}")
        suggestion = Suggestion_Generation_Agent(image_data, labels_data, score_Review_data, suggestion_data, dimension)
        try:
            suggestion_data = {
                "suggestions": {
                    "original": suggestion,
                    "current": suggestion,
                    "added": "",
                    "removed": ""
                }
            }
            # Overwrite the generated suggestions
            with open(suggestion_filepath, 'w') as f:
                json.dump(suggestion_data, f)
            print(f"Successfully saved suggestion file: {suggestion_filepath}")
        except Exception as e:
            print(f"Error saving generated suggestion data: {e}")
            return jsonify({"error": "An error occurred during suggestion saving."}), 500
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred, please try again later."}), 500
@app.route('/update_radar_chart', methods=['POST'])
def update_radar_chart():
    scores = list(global_scores.values())  # Get all current scores
    radar_chart = plot_radar_chart(scores)
    return jsonify({"radar_chart": radar_chart})
def process_text_change(old_text, new_text):
    removed = []
    added = []
    # Use difflib.SequenceMatcher to find the exact changes
    matcher = difflib.SequenceMatcher(None, old_text, new_text)
    # Iterate over the opcodes and identify which parts are replaced, deleted, or added
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Replace operation - the part in old_text is replaced by the part in new_text
            removed.append(old_text[i1:i2])
            added.append(new_text[j1:j2])
        elif tag == 'delete':
            # Delete operation - the part in old_text is deleted
            removed.append(old_text[i1:i2])
        elif tag == 'insert':
            # Insert operation - the part in new_text is added
            added.append(new_text[j1:j2])
    # Concatenate the results into a string
    return ''.join(added), ''.join(removed)
@app.route('/save_user_actions', methods=['POST'])
def save_user_actions():
    data = request.json
    image_name = data.get('image_name', 'default_image')
    user_actions = data.get('user_actions', {})
    print('user_actions',user_actions)
    scores = user_actions.get('scores', {})
    Reviews = user_actions.get('Reviews', {})
    suggestions = user_actions.get('suggestions', {})  # Get suggestions data
    Entities = user_actions.get('Entities', {})
    print("##save##")
    print("Entities",Entities)
    style = user_actions.get('style', {})  # Get style data
    print("Style",style)
    # Define three folders
    score_Review_folder = 'userActions/score_Review'
    suggestion_folder = 'userActions/suggestion'
    Entities_folder = 'userActions/Entities'
    # Creat folders
    os.makedirs(score_Review_folder, exist_ok=True)
    os.makedirs(suggestion_folder, exist_ok=True)
    os.makedirs(Entities_folder, exist_ok=True)
    # Saving scores and reviews data
    if scores or Reviews:
        for dimension in scores.keys():
            score_Review_filename = f"{image_name}_{dimension}_score_Review.json"
            score_Review_filepath = os.path.join(score_Review_folder, score_Review_filename)
            if os.path.exists(score_Review_filepath):
                with open(score_Review_filepath, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {
                    "scores": {"original": 0, "current": 0, "initGPTscore": None},
                    "Reviews": {"original": "", "current": "", "added": "", "removed": ""},
                }
            new_score = scores.get(dimension, {}).get('current', existing_data['scores']['current'])
            new_Review = Reviews.get(dimension, {}).get('current', existing_data['Reviews']['current'])
            # Update original, current, added, removed
            Review_added, Review_removed = process_text_change(existing_data['Reviews']['original'], new_Review)
            existing_data['Reviews']['added'] = Review_added
            existing_data['Reviews']['removed'] = Review_removed
            existing_data['scores']['current'] = new_score
            existing_data['Reviews']['current'] = new_Review
            if existing_data['scores']['original'] == 0:
                existing_data['scores']['original'] = new_score
            if existing_data['Reviews']['original'] == "":
                existing_data['Reviews']['original'] = new_Review
            # Make sure initGPTscore remains unchanged
            if existing_data['scores']['initGPTscore'] is None:
                existing_data['scores']['initGPTscore'] = new_score
            try:
                with open(score_Review_filepath, 'w') as f:
                    json.dump(existing_data, f)
                print(f"Successfully saved score and Review file: {score_Review_filepath}")
            except Exception as e:
                print(f"Failed to save score and Review file: {score_Review_filepath}, Error: {e}")
    # Save suggestion data
    if suggestions:
        for dimension in suggestions.keys():
            suggestion_filename = f"{image_name}_{dimension}_suggestion.json"
            suggestion_filepath = os.path.join(suggestion_folder, suggestion_filename)

            if os.path.exists(suggestion_filepath):
                with open(suggestion_filepath, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {"suggestions": {"original": "", "current": "", "added": "", "removed": ""}}

            new_suggestion = suggestions.get(dimension, {}).get('current', existing_data['suggestions']['current'])

            # Update original, current, added, removed
            suggestion_added, suggestion_removed = process_text_change(existing_data['suggestions']['original'],
                                                                       new_suggestion)
            existing_data['suggestions']['current'] = new_suggestion
            existing_data['suggestions']['added'] = suggestion_added
            existing_data['suggestions']['removed'] = suggestion_removed
            if existing_data['suggestions']['original'] == "":
                existing_data['suggestions']['original'] = new_suggestion
            try:
                with open(suggestion_filepath, 'w') as f:
                    json.dump(existing_data, f)
                print(f"Successfully saved suggestion file: {suggestion_filepath}")
            except Exception as e:
                print(f"Failed to save suggestion file: {suggestion_filepath}, Error: {e}")
    # Save entity data, including style entities
    if Entities.get('added') or Entities.get('removed') or style.get('added') or style.get('removed'):
        labels_filename = f"{image_name}_labels.json"
        labels_filepath = os.path.join(Entities_folder, labels_filename)
        try:
            # Load existing tags and style data, ensuring that style is not modified
            if os.path.exists(labels_filepath):
                with open(labels_filepath, 'r') as f:
                    existing_data = json.load(f)
                style_data = existing_data.get("style", {"original": [], "added": [], "removed": []})
            else:
                style_data = {"original": [], "added": [], "removed": []}
            # Update normal entities
            data_to_save = {"original": Entities.get('original', []), "added": Entities.get('added', []), "removed": Entities.get('removed', []), "style": style}
            # Save updated data
            with open(labels_filepath, 'w') as f:
                json.dump(data_to_save, f)
            print(f"Successfully saved labels and style to file: {labels_filepath}")
        except Exception as e:
            print(f"Failed to save labels and style to file: {labels_filepath}, Error: {e}")

    return jsonify({"status": "success"})
@app.route('/submit_score_Review', methods=['POST'])
def submit_score_Review():
    data = request.json
    image_name = data['image_name']
    dimension = data['dimension']
    # Check that the dimension is passed correctly
    print(f"Dimension received: {dimension}")

    try:
        score_Review_filepath = os.path.join('userActions/score_Review',
                                              f"{image_name}_{dimension}_score_Review.json")
        if os.path.exists(score_Review_filepath):
            with open(score_Review_filepath, 'r') as f:
                score_Review_data = json.load(f)
        else:
            score_Review_data = {
            "scores": {"original": 0, "current": 0, "initGPTscore": None},
            "Reviews": {"original": "", "current": "", "added": "", "removed": ""},
        }
    except Exception as e:
        print(f"Error loading score/Review data: {e}")
        score_Review_data = {
            "scores": {"original": 0, "current": 0, "initGPTscore": None},
            "Reviews": {"original": "", "current": "", "added": "", "removed": ""},
        }
    save_round_data(image_name, dimension, 'score_Review', score_Review_data)

    return jsonify({"status": "success", "message": f"Score and Review for {dimension} submitted successfully"})
@app.route('/submit_suggestion', methods=['POST'])
def submit_suggestion():
    data = request.json
    image_name = data['image_name']
    dimension = data['dimension']

    # Check that the dimension is passed correctly
    print(f"Dimension received: {dimension}")

    try:
        suggestion_filepath = os.path.join('userActions/suggestion',
                                              f"{image_name}_{dimension}_suggestion.json")
        if os.path.exists(suggestion_filepath):
            with open(suggestion_filepath, 'r') as f:
                suggestion_data = json.load(f)
        else:
            suggestion_data = {"suggestions": {"original": "","current": "","added": "","removed": ""}}
    except Exception as e:
        print(f"Error loading score/Review data: {e}")
        suggestion_data = {"suggestions": {"original": "","current": "","added": "","removed": ""}}

    # Call the save_round_data function to save the last round of suggestion data
    save_round_data(image_name, dimension, 'suggestion', suggestion_data)

    return jsonify({"status": "success", "message": f"Suggestion for {dimension} submitted successfully"})
# Save the rating and recommendation data for each round and calculate the round independently
def save_round_data(image_name, dimension, data_type, new_data):
    # Define different folders
    folder = os.path.join('userActionsEveryRounds', data_type)
    os.makedirs(folder, exist_ok=True)
    # Initialize or increase the round counter to ensure that score_Review and suggestion are calculated independently
    if image_name not in round_counters:
        round_counters[image_name] = {'score_Review': {}, 'suggestion': {}}
    # Ensure that score_Review and suggestion rounds are independent
    round_counters[image_name][data_type].setdefault(dimension, 0)
    round_counters[image_name][data_type][dimension] += 1
    current_round = round_counters[image_name][data_type][dimension]
    print(f"Current round for {dimension} ({data_type}): {current_round}")
    # Build file name
    round_filename = f"{image_name}_{dimension}_{data_type}.json"
    round_filepath = os.path.join(folder, round_filename)
    print(f"Saving round data for {dimension} ({data_type}) at {round_filepath}")
    # Add debugging information to print the path and content of saved data
    print(f"Saving round data to {round_filepath}")
    print(f"Current round: {current_round}")
    print(f"Data to be saved: {new_data}")
    # Read previous data and merge
    if os.path.exists(round_filepath):
        with open(round_filepath, 'r') as f:
            previous_rounds_data = json.load(f)
    else:
        previous_rounds_data = []
    print(f"Previous rounds data for {dimension}: {previous_rounds_data}")
    # Add new data for the current round
    previous_rounds_data.append({'round': current_round, 'data': new_data})

    # Save the merged data to a file
    with open(round_filepath, 'w') as f:
        json.dump(previous_rounds_data, f)

    print(f"Successfully saved round data for {dimension} in round {current_round}")
# Draw radar chart
def plot_radar_chart(scores):
    labels = np.array([
        'Realism', 'Deformation', 'Imagination', 'Color Richness', 'Color Contrast',
        'Line Combination', 'Line Texture', 'Picture Organization', 'Transformation'
    ])
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The radar chart is a circle, so it needs to be "closed".
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='red', alpha=0.25)
    ax.plot(angles, scores, color='red', linewidth=2)
    plt.xticks(angles[:-1], labels, fontsize=12)  # Set the axis font size

    # Set the range of the radar chart
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=12)  # Set the font size of the tick labels

    # Save the plot as a PNG image with higher DPI for better clarity
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)  # You can adjust DPI based on your requirement
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)  # Close the figure to free memory
    return base64_image
if __name__ == '__main__':
    app.run(debug=True)