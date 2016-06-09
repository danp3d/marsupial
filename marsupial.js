'use strict';

const marsupial_native = require('./build/Release/marsupial.node')
const Promise = require('bluebird')

module.exports = {
    trainObjectDetector: (data, outputDetectorName) => new Promise((resolve, reject) => {
        return marsupial_native.trainObjectDetector(data, outputDetectorName, (err) => {
            if (err) return reject(err)

            return resolve(null)
        })
    }),

    detectObjects: (imageFileName, detectorFileName) => new Promise((resolve, reject) => {
        return marsupial_native.detectObjects(imageFileName, detectorFileName, (err, results) => {
            if (err) return reject(err)

            return resolve(results)
        })
    })
}

