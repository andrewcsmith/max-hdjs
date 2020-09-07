// import * as hdjs from 'hd-js'
// import * as Max from 'max-api'
const hdjs = require('hd-js')
const tf = require('@tensorflow/tfjs-node-gpu')
const Max = require('max-api')

const callback = async function() {
  let pitches = await minimizer.logPitches.array()
  Max.outlet(pitches[0])
}

let minimizer = new hdjs.Minimizer({
  primeLimits: [19, 12, 2, 2, 1, 1],
  dimensions: 3,
  bounds: [-4.0, 4.0],
  callback: callback,
})

Max.post("Loaded the script")

minimizer.vs.init().then(() => {
  Max.post("Successfully initialized minimizer VectorSpace")
}).catch(err => {
  Max.post(err, Max.POST_LEVELS.ERROR)
})

const handlers = {
  setLogPitches: (...pitches) => {
    let accum = []
    for (pitch of pitches) {
      accum.push(pitch)
    }
    minimizer.logPitches.assign(tf.tensor([accum]))
  },

  setLogPitch: async (idx, pitch) => {
    let currentPitches = await minimizer.logPitches.array()
    currentPitches[0][idx] = pitch
    minimizer.logPitches.assign(tf.tensor(currentPitches))
  },

  getWeights: async () => {
    let weights = await minimizer.opt.getWeights()
    for (weight of weights) {
      Max.post(await weight.tensor.array())
    }
  },

  reinitializeWeights: async () => {
    await minimizer.reinitializeWeights()
  },

  [Max.MESSAGE_TYPES.BANG]: () => {
    minimizer.takeStep().then(() => {
      // Max.post("successful step", Max.POST_LEVELS.INFO)
    }).catch(err => {
      Max.post(err, Max.POST_LEVELS.ERROR)
    })
  }
}

Max.addHandlers(handlers)
