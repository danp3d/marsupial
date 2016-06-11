'use strict'

const fs = require('fs')
const path = require('path')
const Promise = require('bluebird')
const exec = require('child_process').exec

/* eslint no-console:0 */
if (fs.existsSync(path.resolve(__dirname, 'build'))) {
	console.log('DLib already built')
	process.exit(0)
}

const execCmd = (cmd, dir) => new Promise((resolve, reject) => {
	return exec(cmd, {
		cwd: path.resolve(__dirname, dir),
		maxBuffer: 1024 * 1024
	}, (err, stdout, stderr) => {
		if (err) return reject(err)

		return resolve({
			stdout: stdout,
			stderr: stderr
		})
	})
})

fs.mkdirSync(path.resolve(__dirname, 'dlib', 'build'))
execCmd('cmake ..', 'dlib/build').then((res) => {
	console.log(res.stdout)
	return execCmd('make', 'dlib/build')
}).then((res) => {
	console.log(res.stdout)
	if (!fs.existsSync(path.resolve(__dirname, 'dlib', 'build', 'libdlib.so'))) {
		throw new Error('Could not find file libdlib.so, even though build was successfull')
	}

	process.exit(0)
}).catch((err) => {
	console.log('ERROR:', err.message, err.stack, err.code, err)
	process.exit(1)
})
