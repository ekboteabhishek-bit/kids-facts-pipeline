/**
 * Remotion Effects Renderer
 * 
 * Takes a video + effects config and renders the final video with effects applied.
 * Called from the Python backend.
 * 
 * Usage:
 *   node render-effects.js --config effects-config.json --output output.mp4
 */

const { bundle } = require('@remotion/bundler');
const { renderMedia, selectComposition } = require('@remotion/renderer');
const path = require('path');
const fs = require('fs');

async function renderWithEffects(configPath, outputPath) {
  console.log('🎬 Starting Remotion render...');
  
  // Load effects config
  const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  
  console.log(`📁 Video source: ${config.videoSrc}`);
  console.log(`🎨 Effects: ${config.effects.length}`);
  console.log(`💬 Captions: ${config.captions.length}`);
  
  // Bundle the Remotion project
  console.log('📦 Bundling...');
  const bundleLocation = await bundle({
    entryPoint: path.join(__dirname, 'src/index.jsx'),
    webpackOverride: (config) => config,
  });
  
  // Select the composition
  const composition = await selectComposition({
    serveUrl: bundleLocation,
    id: 'KidsFactsVideo',
    inputProps: {
      videoSrc: config.videoSrc,
      audioSrc: config.audioSrc || null,
      effects: config.effects || [],
      captions: config.captions || [],
      captionStyle: config.captionStyle || {}
    }
  });
  
  // Override duration if specified
  const finalComposition = {
    ...composition,
    durationInFrames: config.durationInFrames || composition.durationInFrames
  };
  
  // Render
  console.log('🎥 Rendering...');
  await renderMedia({
    composition: finalComposition,
    serveUrl: bundleLocation,
    codec: 'h264',
    outputLocation: outputPath,
    inputProps: {
      videoSrc: config.videoSrc,
      audioSrc: config.audioSrc || null,
      effects: config.effects || [],
      captions: config.captions || [],
      captionStyle: config.captionStyle || {}
    },
    onProgress: ({ progress }) => {
      process.stdout.write(`\r   Progress: ${Math.round(progress * 100)}%`);
    }
  });
  
  console.log('\n✅ Render complete!');
  console.log(`📁 Output: ${outputPath}`);
}

// Parse command line arguments
const args = process.argv.slice(2);
let configPath = null;
let outputPath = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--config' && args[i + 1]) {
    configPath = args[i + 1];
  }
  if (args[i] === '--output' && args[i + 1]) {
    outputPath = args[i + 1];
  }
}

if (!configPath || !outputPath) {
  console.log('Usage: node render-effects.js --config <config.json> --output <output.mp4>');
  process.exit(1);
}

renderWithEffects(configPath, outputPath)
  .catch(err => {
    console.error('❌ Render failed:', err);
    process.exit(1);
  });
