# Marsupial
Very simple pattern recognition tool for node.js (written as a native C++ addon), using DLib.

## Usage
**IMPORTANT: You need the basic build tools and libjpeg installed**  
on Ubuntu, do
```
    $ sudo apt install build-essential libjpeg-dev
```
### Developing
```
    git clone https://github.com/danp3d/marsupial.git
    cd marsupial
    npm install
```

### Using in a node project
```
    npm install --save git+https://github.com/danp3d/marsupial.git
```
```javascript
    const marsupial = require('marsupial')

    // Training an object detector
    marsupial.trainObjectDetector(
        [
            {
                "imageFileName": "data/images/image1.jpg",
                "machAreas": [{ // Rectangles with the area that the object detector should use - dimensions in pixels
                    "top": "3",
                    "left": "5",
                    "width": "200",
                    "height": "200"
                }]
            }
        ],
        "data/objectDetector1.svm"
    ).then(() => {
        console.log("Successfully trained!")    
    })

    // Using an object detector
    marsupial.detectObjects("data/images/image1.jpg", "data/objectDetector1.svm").then((matches) => {
        console.log("Found", matches.length, "matches")
        console.log("First area: { ",
            "top:", matches[0].top, 
            ", left: ", matches[0].left, 
            ", width: ", matches[0].width, 
            ", height: ", matches[0].height, 
        "}")
    })
```


