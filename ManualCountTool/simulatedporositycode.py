#This code is for the creation of simulated porosity images of size 512x512 pixels
#These simulated images have a set porosity, size distribution, and orientation with a random grayscale fill color
#Parameters to change are in the main function
import math
import random
import numpy as np
from PIL import Image, ImageDraw

def ellipse_area(radius_x, radius_y):
    area = math.pi*radius_x*radius_y
    return area

def random_radius(min_radius, max_radius):
    radius_x = random.randint(min_radius, max_radius) #random radius values
    radius_y = random.randint(min_radius, max_radius)
    return radius_x, radius_y

def random_position(image_size_x, image_size_y):
    center_x = random.randint(0, image_size_x) #random position for x
    center_y = random.randint(0, image_size_y) #random position for y
    return center_x, center_y

def draw_image(image_size_x, image_size_y):
    img = Image.new('RGBA', (image_size_x, image_size_y), color=0) #creates image for plotting
    return img

def small(porosity_area, image_size_x, image_size_y):
    min_radius = 2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 34/2 #units: pixels (chosen based on small pores being smaller than 6.4 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def medium(porosity_area, image_size_x, image_size_y):
    min_radius = 36/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 50 #units: pixels (chosen based on medium pores being between 6.4 and 16.2 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def large(porosity_area, image_size_x, image_size_y):
    min_radius = 52/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 75 #units: pixels (arbitrarily chosen so that pores aren't ridiculously large)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def mixed(porosity_area, image_size_x, image_size_y):
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        size = random.randint(1, 3)
        if size == 1: #small pore
            min_radius = 2
            max_radius = 34/2
        elif size == 2: #medium pore
            min_radius = 36/2
            max_radius = 50
        elif size == 3: #large pore
            min_radius = 52/2
            max_radius = 100
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def smallclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    min_radius = 2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 34/2 #units: pixels (chosen based on small pores being smaller than 6.4 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def mediumclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    min_radius = 36/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 50 #units: pixels (chosen based on medium pores being between 6.4 and 16.2 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def largeclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    min_radius = 52/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 75 #units: pixels (arbitrarily chosen so that pores aren't ridiculously large)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

def mixedclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while sum(total_area) <= porosity_area:
        size = random.randint(1, 3)
        if size == 1: #small pore
            min_radius = 2
            max_radius = 34/2
        elif size == 2: #medium pore
            min_radius = 36/2
            max_radius = 50
        elif size == 3: #large pore
            min_radius = 52/2
            max_radius = 100
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(50, 255) #gives random grayscale value to fill each ellipse (no white values allowed)
        rgba_value = ((grayscale_value, grayscale_value, grayscale_value,255))
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=rgba_value, outline=(255,255,255,255))
    img_array = np.array(img)
    return Image.fromarray(img_array)

if __name__ == "__main__":
    #set desired number of images, porosity, image size, size distribution, and orientation
    num_images = 1
    set_porosity = 50 #units: percent
    image_size_x = 512 #units: pixels
    image_size_y = 512 #units: pixels
    porosity_area = (set_porosity/100)*image_size_x * image_size_y #units: pixels^2
    #options for size_dist are: 'small', 'medium', 'large', and 'mixed' (for random number of all sizes)
    size_dist = 'large'
    #options for orientation are: 'random' or 'clustered' (for pores clustered in one area of the image)
    orientation = 'random'

    if orientation == 'random':
        if size_dist == 'small':
            for i in range(num_images):
                random_image = small(porosity_area, image_size_x, image_size_y)
                random_image.save(f"random_image{i+1}.png")
        elif size_dist == 'medium':
            for i in range(num_images):
                random_image = medium(porosity_area, image_size_x, image_size_y)
                random_image.save(f"random_image{i+1}.png")
        elif size_dist == 'large':
            for i in range(num_images):
                random_image = large(porosity_area, image_size_x, image_size_y)
                random_image.save(f"random_image{i+1}.png")
        elif size_dist == 'mixed':
            for i in range(num_images):
                random_image = mixed(porosity_area, image_size_x, image_size_y)
                random_image.save(f"random_image{i+1}.png")
    elif orientation == 'clustered':
        #this creates a random size bounding box for pore placement on each image
        boundingbox_x = random.randint(1, image_size_x)
        boundingbox_y = random.randint(1, image_size_y)
        if size_dist == 'small':
            for i in range(num_images):
                random_image = smallclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
                random_image.save(f"random_image{i+1}.png")
        elif size_dist == 'medium':
            for i in range(num_images):
                random_image = mediumclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
                random_image.save(f"random_image{i+1}.png")
        elif size_dist == 'large':
            for i in range(num_images):
                random_image = largeclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
                random_image.save(f"random_image{i+1}.png")
        elif size_dist == 'mixed':
            for i in range(num_images):
                random_image = mixedclustered(porosity_area, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
                random_image.save(f"random_image{i+1}.png")

