'use strict';

const fs = require('fs')
const path = require('path')
const should = require('should')
const marsupial_native = require('../build/Release/marsupial.node')
const marsupial = require('../marsupial')

const outputPath = path.resolve(__dirname, 'output')
const objectDetectorName = path.resolve(outputPath, 'object_detector.svm')
const testImageName = path.resolve(__dirname, 'fixtures', 'to_test.jpg')
const trainingData = require('./fixtures/trainingData.json').map((record) => {
    record.imageFileName = path.resolve(__dirname, record.imageFileName)
    return record
})

require('mocha-sinon');

describe('Marsupial', () => {
    before(() => {
        if (!fs.existsSync(outputPath))
            fs.mkdirSync(outputPath)

        if (fs.existsSync(objectDetectorName))
            fs.unlinkSync(objectDetectorName)
    })

    it('should train an object detector', function (done) {
        this.enableTimeouts(false)

        marsupial.trainObjectDetector(trainingData, objectDetectorName)
            .then(() => {
                done()
            })
            .catch(done)
    })

    it('should detect the test image', (done) => {
        marsupial.detectObjects(testImageName, objectDetectorName)
            .then((detected) => {
                detected.should.be.ok()
                detected.length.should.equal(1)
                detected[0].top.should.be.within(120, 140)
                detected[0].left.should.be.within(390, 405)
                detected[0].width.should.be.within(210, 225)
                detected[0].height.should.be.within(210, 225)
                done()
            })
            .catch(done)
    })

    it('should handle errors', function (done) {
        this.sinon.stub(marsupial_native, 'trainObjectDetector', (a, b, c) => c('error'))
        this.sinon.stub(marsupial_native, 'detectObjects', (a, b, c) => c('error'))

        marsupial.trainObjectDetector('a', 'b').then(() => done('Oops. Did not throw')).catch((err) => {
            err.should.equal('error')
            marsupial.detectObjects('a', 'b').then(() => done('Oops. Did not throw')).catch((err) => {
                err.should.equal('error')
                done()
            })
        })
    })
})
