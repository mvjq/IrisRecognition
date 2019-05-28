# Iris Recognition
Abstract: This project intends to identify humans by they iris using techniques of image processing. Given a image of a face and eyes, return a validation of that person (if is not in database, we'll insert it for future references).

### **Datasets**:

- *Upol*: http://phoenix.inf.upol.cz/iris/ 
- *MMU*: https://www.cs.princeton.edu/~andyz/downloads/MMUIrisDatabase.zip    
- *CASIA1*: http://www.cbsr.ia.ac.cn/IrisDatabase.htm 
  
  
### **Image Examples:**:

###### UPOL Image Example 
<p align='center'>
  <img src='https://i.ibb.co/SxZXSTv/001L-1.png' />  
</p>


###### MMU Image Example
<p align='center'>
  <img src='https://i.ibb.co/qjXNHzR/norazal5.png' />
</p>

###### CASIA1 Image Example

<p align='center'>
    <img src='https://i.ibb.co/nmqb4pv/084-2-2.jpg' />
</p>


### **Process:**

1) Process the dataset, make the images more easy to process (there's a possibility to remove eyelashes to earn a more precise image but first, we'll guarantee that the methods here work without worrying about this )
2) We  extract the pupils and iris using a couple of techniques (integrodifferential and Hough transform)
3) We normalize the extracted iris to make the comparison possible 
4) In this part, we verify the image using a couple of filters (Gabor/Log-Gabor and, Laplacian of Gaussian)
5) Here it's the comparison part: we check in the database for an image equals/similar to the iris processed. We use Hamming Distance, Euclidian Distance and, Norm Correlation)
6) If the image is in the database, we validate it. If not, there's a possibility to insert for future verifications.

### **Source Code**:
