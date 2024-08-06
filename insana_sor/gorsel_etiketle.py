import os
from PIL import Image, ImageDraw, ImageFont


def label_boxes_on_all_images(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Load the image
            img = Image.open(input_path)
            width, height = img.size

            # Create a drawing context
            draw = ImageDraw.Draw(img)

            # Define the text properties
            font = ImageFont.load_default()
            text_color = (255, 0, 0)  # Red color

            # Define the positions for the text (specified coordinates)
            pos_1 = (380, 635)
            pos_2 = (380, 830)
            pos_3 = (1150, 635)
            pos_4 = (1150, 830)

            # Draw the text on the image
            draw.text(pos_1, "1", font=font, fill=text_color)
            draw.text(pos_2, "2", font=font, fill=text_color)
            draw.text(pos_3, "3", font=font, fill=text_color)
            draw.text(pos_4, "4", font=font, fill=text_color)

            # Save the modified image
            img.save(output_path)

    print("All images have been processed and saved successfully.")


# Directory containing the input images
input_directory = "anonim_gorseller"

# Directory to save the labeled images
output_directory = "numarali_gorseller"

# Call the function
label_boxes_on_all_images(input_directory, output_directory)
