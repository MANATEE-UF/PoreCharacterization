#This code is for the creation of simulated porosity images of size 512x512 pixels
#These simulated images have a set porosity, size distribution, and orientation with a random grayscale fill color
#Parameters to change are in the main function
import math
import random
import numpy as np
from PIL import Image, ImageDraw
import os

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
    img = Image.new('L', (image_size_x, image_size_y), color=255) #creates image for plotting
    return img

def small(targetPorosity, image_size_x, image_size_y):
    min_radius = 2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 34/2 #units: pixels (chosen based on small pores being smaller than 6.4 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(0, 150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def medium(targetPorosity, image_size_x, image_size_y):
    min_radius = 36/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 50 #units: pixels (chosen based on medium pores being between 6.4 and 16.2 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def large(targetPorosity, image_size_x, image_size_y):
    min_radius = 26 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 75 #units: pixels (arbitrarily chosen so that pores aren't ridiculously large)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def mixed(targetPorosity, image_size_x, image_size_y):
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
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
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(image_size_x, image_size_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def smallclustered(targetPorosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    min_radius = 2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 34/2 #units: pixels (chosen based on small pores being smaller than 6.4 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def mediumclustered(targetPorosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    min_radius = 36/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 50 #units: pixels (chosen based on medium pores being between 6.4 and 16.2 microns in diameter in literature)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def largeclustered(targetPorosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    min_radius = 52/2 #units: pixels (chosen to allow to be seen using manual counting program)
    max_radius = 75 #units: pixels (arbitrarily chosen so that pores aren't ridiculously large)
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
        radius_x, radius_y = random_radius(min_radius, max_radius)
        area = ellipse_area(radius_x, radius_y)
        total_area.append(area)
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def mixedclustered(targetPorosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y):
    total_area = []
    drawn_positions = []
    img = draw_image(image_size_x, image_size_y)
    draw = ImageDraw.Draw(img)
    while porositycalc(np.array(img), image_size_x, image_size_y) <= targetPorosity:
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
        grayscale_value = random.randint(0,150) #gives random grayscale value to fill each ellipse (no white values allowed)
        center_x, center_y = random_position(boundingbox_x, boundingbox_y)
        drawn_positions.append((center_x, center_y, max(radius_x, radius_y))) #save where each ellipse is drawn
        draw.ellipse([(center_x - radius_x, center_y - radius_y),
                    (center_x + radius_x, center_y + radius_y)],
                    fill=grayscale_value, outline=0)
    img_array = np.array(img)
    real_porosity = porositycalc(img_array, image_size_x, image_size_y)
    return Image.fromarray(img_array), real_porosity

def porositycalc(img_array, x, y):
    img_area = x*y
    img_porosity = (np.sum(img_array!=255)/img_area) * 100
    return img_porosity

def main(set_porosity, size_dist, orientation):
    #set desired number of images, porosity, image size, size distribution, and orientation
    num_images = 1
    image_size_x = 1024 #units: pixels
    image_size_y = 1024 #units: pixels
    #options for size_dist are: 'small', 'medium', 'large', and 'mixed' (for random number of all sizes)
    #options for orientation are: 'random' or 'clustered' (for pores clustered in one area of the image)

    dirName = f"Case_{set_porosity}_{size_dist}_{orientation}"
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    bbox_min = np.sqrt(set_porosity/100) # ensures that bbox is large enough to fit enough pores to meet porosity. assumes square

    for i in range(num_images):
        if orientation == 'random':
            if size_dist == 'small':
                random_image, real_porosity = small(set_porosity, image_size_x, image_size_y)
            elif size_dist == 'medium':
                random_image, real_porosity = medium(set_porosity, image_size_x, image_size_y)
            elif size_dist == 'large':
                random_image, real_porosity = large(set_porosity, image_size_x, image_size_y)
            elif size_dist == 'mixed':
                random_image, real_porosity = mixed(set_porosity, image_size_x, image_size_y)
        elif orientation == 'clustered':
            #this creates a random size bounding box for pore placement on each image
            boundingbox_x = random.randint(int(image_size_x*bbox_min), image_size_x)
            boundingbox_y = random.randint(int(image_size_y*bbox_min), image_size_y)
            if size_dist == 'small':
                random_image, real_porosity = smallclustered(set_porosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
            elif size_dist == 'medium':
                random_image, real_porosity = mediumclustered(set_porosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
            elif size_dist == 'large':
                random_image, real_porosity = largeclustered(set_porosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
            elif size_dist == 'mixed':
                random_image, real_porosity = mixedclustered(set_porosity, image_size_x, image_size_y, boundingbox_x, boundingbox_y)
        
        random_image.save(f"{dirName}/image{i+1}_actualporosity{real_porosity:.2f}.png")
    

if __name__ == "__main__":
    porosityVals = [5, 12.5, 25, 37.5, 50]
    poreSizes = ["small", "medium", "large", "mixed"]
    distributions = ["random", "clustered"]
    for i in range(len(porosityVals)):
        for j in range(len(poreSizes)):
            for k in range(len(distributions)):
                main(porosityVals[i], poreSizes[j], distributions[k])