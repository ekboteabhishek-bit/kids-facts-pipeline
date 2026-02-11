import { AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig, interpolate, spring, Video, Audio, Img } from 'remotion';
import React from 'react';

// ============== EFFECT COMPONENTS ==============

/**
 * Zoom Pulse Effect
 * Adds a subtle zoom pulse on key moments
 */
export const ZoomPulse = ({ children, triggerFrame, intensity = 1.1, duration = 15 }) => {
  const frame = useCurrentFrame();
  
  const scale = interpolate(
    frame,
    [triggerFrame, triggerFrame + duration / 2, triggerFrame + duration],
    [1, intensity, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
    <div style={{ transform: `scale(${scale})`, width: '100%', height: '100%' }}>
      {children}
    </div>
  );
};

/**
 * Screen Shake Effect
 * Adds camera shake for impact moments (BOOM!)
 */
export const ScreenShake = ({ children, triggerFrame, intensity = 10, duration = 12 }) => {
  const frame = useCurrentFrame();
  
  if (frame < triggerFrame || frame > triggerFrame + duration) {
    return <>{children}</>;
  }
  
  const progress = (frame - triggerFrame) / duration;
  const decay = 1 - progress;
  
  // Random-ish shake using frame number as seed
  const shakeX = Math.sin(frame * 47) * intensity * decay;
  const shakeY = Math.cos(frame * 53) * intensity * decay;
  const rotation = Math.sin(frame * 31) * (intensity / 5) * decay;

  return (
    <div style={{
      transform: `translate(${shakeX}px, ${shakeY}px) rotate(${rotation}deg)`,
      width: '100%',
      height: '100%'
    }}>
      {children}
    </div>
  );
};

/**
 * Emoji Pop Effect
 * Animated emoji that pops in and floats
 */
export const EmojiPop = ({ emoji, x, y, triggerFrame, duration = 30 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  
  if (frame < triggerFrame || frame > triggerFrame + duration) {
    return null;
  }
  
  const localFrame = frame - triggerFrame;
  
  const scale = spring({
    frame: localFrame,
    fps,
    config: { damping: 10, stiffness: 100 }
  });
  
  const opacity = interpolate(
    localFrame,
    [0, 5, duration - 10, duration],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  
  const floatY = interpolate(localFrame, [0, duration], [0, -30]);

  return (
    <div style={{
      position: 'absolute',
      left: `${x}%`,
      top: `${y}%`,
      transform: `scale(${scale}) translateY(${floatY}px)`,
      opacity,
      fontSize: 60,
      zIndex: 100
    }}>
      {emoji}
    </div>
  );
};

/**
 * Kinetic Text Effect
 * Supports two modes:
 * - "karaoke": Full sentence visible, current word highlighted progressively
 * - "word_by_word": Words pop in one at a time (legacy)
 */
export const KineticText = ({
  text,
  startFrame,
  wordsPerSecond = 3,
  highlightWords = [],
  highlightColor = '#4A90A4',
  captionStyle = 'karaoke',
  bgStyle = 'dark',
  style = {}
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const words = text.split(' ');
  const framesPerWord = fps / wordsPerSecond;

  if (captionStyle === 'karaoke') {
    // Karaoke mode: all words visible, current word highlighted
    const totalFrames = words.length * framesPerWord;

    // Fade in the whole sentence
    const sentenceOpacity = interpolate(
      frame,
      [startFrame, startFrame + 6],
      [0, 1],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );

    return (
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        gap: '6px',
        opacity: sentenceOpacity,
        ...style
      }}>
        {words.map((word, index) => {
          const wordFrame = startFrame + index * framesPerWord;
          const isSpoken = frame >= wordFrame;
          const isCurrentWord = frame >= wordFrame && frame < wordFrame + framesPerWord;

          // Scale bounce on the current word
          const localFrame = frame - wordFrame;
          const currentScale = isCurrentWord ? spring({
            frame: localFrame,
            fps,
            config: { damping: 15, stiffness: 180 }
          }) : 1;

          const wordScale = isCurrentWord ? 1.0 + (currentScale - 1) * 0.12 : 1;

          const bgAlpha = bgStyle === 'dark' ? '33' : '00';
          const wordBg = isSpoken ? `${highlightColor}${bgAlpha}` : 'transparent';

          return (
            <span
              key={index}
              style={{
                display: 'inline-block',
                transform: `scale(${wordScale})`,
                color: isSpoken ? highlightColor : 'rgba(255,255,255,0.45)',
                fontWeight: isSpoken ? 800 : 600,
                fontSize: isCurrentWord ? '1.05em' : '1em',
                background: wordBg,
                padding: '2px 5px',
                borderRadius: '4px',
                transition: 'color 0.05s',
              }}
            >
              {word}
            </span>
          );
        })}
      </div>
    );
  }

  // Word-by-word pop-in mode (legacy)
  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      gap: '8px',
      ...style
    }}>
      {words.map((word, index) => {
        const wordStartFrame = startFrame + index * framesPerWord;
        const isVisible = frame >= wordStartFrame;
        const isHighlight = highlightWords.some(hw =>
          word.toLowerCase().includes(hw.toLowerCase())
        );

        const localFrame = frame - wordStartFrame;

        const scale = isVisible ? spring({
          frame: localFrame,
          fps,
          config: { damping: 12, stiffness: 200 }
        }) : 0;

        const opacity = isVisible ? interpolate(
          localFrame,
          [0, 5],
          [0, 1],
          { extrapolateRight: 'clamp' }
        ) : 0;

        return (
          <span
            key={index}
            style={{
              display: 'inline-block',
              transform: `scale(${scale})`,
              opacity,
              color: isHighlight ? highlightColor : 'white',
              fontWeight: isHighlight ? 800 : 700,
              fontSize: isHighlight ? '1.1em' : '1em'
            }}
          >
            {word}
          </span>
        );
      })}
    </div>
  );
};

/**
 * Flash Effect
 * Quick white flash for impact
 */
export const FlashEffect = ({ triggerFrame, duration = 6, color = 'white' }) => {
  const frame = useCurrentFrame();
  
  if (frame < triggerFrame || frame > triggerFrame + duration) {
    return null;
  }
  
  const opacity = interpolate(
    frame,
    [triggerFrame, triggerFrame + 2, triggerFrame + duration],
    [0, 0.8, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
    <AbsoluteFill style={{
      backgroundColor: color,
      opacity,
      zIndex: 50
    }} />
  );
};

/**
 * Number Counter Effect
 * Animated counting number
 */
export const NumberCounter = ({ 
  from, 
  to, 
  startFrame, 
  duration = 30,
  prefix = '',
  suffix = '',
  style = {}
}) => {
  const frame = useCurrentFrame();
  
  if (frame < startFrame) {
    return <span style={style}>{prefix}{from}{suffix}</span>;
  }
  
  const progress = Math.min((frame - startFrame) / duration, 1);
  const eased = 1 - Math.pow(1 - progress, 3); // Ease out cubic
  const current = Math.round(from + (to - from) * eased);
  
  return <span style={style}>{prefix}{current.toLocaleString()}{suffix}</span>;
};

/**
 * Particle Burst Effect
 * Particles exploding outward
 */
export const ParticleBurst = ({ 
  triggerFrame, 
  x = 50, 
  y = 50, 
  particleCount = 12,
  duration = 30,
  colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1']
}) => {
  const frame = useCurrentFrame();
  
  if (frame < triggerFrame || frame > triggerFrame + duration) {
    return null;
  }
  
  const localFrame = frame - triggerFrame;
  const particles = [];
  
  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const speed = 3 + Math.random() * 2;
    const distance = localFrame * speed;
    
    const particleX = x + Math.cos(angle) * distance;
    const particleY = y + Math.sin(angle) * distance;
    
    const opacity = interpolate(
      localFrame,
      [0, duration * 0.7, duration],
      [1, 1, 0],
      { extrapolateRight: 'clamp' }
    );
    
    const scale = interpolate(
      localFrame,
      [0, duration],
      [1, 0.3],
      { extrapolateRight: 'clamp' }
    );
    
    particles.push(
      <div
        key={i}
        style={{
          position: 'absolute',
          left: `${particleX}%`,
          top: `${particleY}%`,
          width: 10,
          height: 10,
          borderRadius: '50%',
          backgroundColor: colors[i % colors.length],
          transform: `scale(${scale})`,
          opacity,
          zIndex: 60
        }}
      />
    );
  }
  
  return <>{particles}</>;
};

// ============== MAIN COMPOSITION ==============

/**
 * EffectsLayer Component
 * Wraps video content and applies effects based on config
 */
export const EffectsLayer = ({
  videoSrc,
  audioSrc,
  effects = [],
  captions = [],
  captionStyle = {},
  sfx = []
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  
  // Find active effects for current frame
  const activeZoomPulses = effects.filter(e => e.type === 'zoom_pulse');
  const activeShakes = effects.filter(e => e.type === 'screen_shake');
  const activeFlashes = effects.filter(e => e.type === 'flash');
  const emojis = effects.filter(e => e.type === 'emoji');
  const particles = effects.filter(e => e.type === 'particles');
  
  // Find active caption
  const activeCaption = captions.find(c => 
    frame >= c.startFrame && frame < c.endFrame
  );
  
  // Wrap content in effects
  let content = (
    <AbsoluteFill>
      <Video src={videoSrc} />
    </AbsoluteFill>
  );
  
  // Apply zoom pulses
  activeZoomPulses.forEach(effect => {
    content = (
      <ZoomPulse 
        triggerFrame={effect.frame} 
        intensity={effect.intensity || 1.1}
        duration={effect.duration || 15}
      >
        {content}
      </ZoomPulse>
    );
  });
  
  // Apply screen shakes
  activeShakes.forEach(effect => {
    content = (
      <ScreenShake 
        triggerFrame={effect.frame} 
        intensity={effect.intensity || 10}
        duration={effect.duration || 12}
      >
        {content}
      </ScreenShake>
    );
  });

  return (
    <AbsoluteFill style={{ backgroundColor: 'black' }}>
      {content}
      
      {/* Flash effects */}
      {activeFlashes.map((effect, i) => (
        <FlashEffect 
          key={`flash-${i}`}
          triggerFrame={effect.frame}
          duration={effect.duration || 6}
          color={effect.color || 'white'}
        />
      ))}
      
      {/* Emoji pops */}
      {emojis.map((effect, i) => (
        <EmojiPop
          key={`emoji-${i}`}
          emoji={effect.emoji}
          x={effect.x || 50}
          y={effect.y || 50}
          triggerFrame={effect.frame}
          duration={effect.duration || 30}
        />
      ))}
      
      {/* Particle bursts */}
      {particles.map((effect, i) => (
        <ParticleBurst
          key={`particles-${i}`}
          triggerFrame={effect.frame}
          x={effect.x || 50}
          y={effect.y || 50}
          particleCount={effect.count || 12}
          duration={effect.duration || 30}
        />
      ))}
      
      {/* Captions */}
      {activeCaption && (
        <AbsoluteFill style={{
          justifyContent: 'flex-end',
          alignItems: 'center',
          paddingBottom: captionStyle.bottom || 180
        }}>
          <KineticText
            text={activeCaption.text}
            startFrame={activeCaption.startFrame}
            highlightWords={activeCaption.highlights || []}
            highlightColor={captionStyle.highlightColor || '#8b5cf6'}
            captionStyle={captionStyle.captionMode || 'karaoke'}
            bgStyle={captionStyle.bgStyle || 'dark'}
            style={{
              fontFamily: captionStyle.font || 'Nunito, sans-serif',
              fontSize: captionStyle.fontSize || 64,
              fontWeight: 700,
              textShadow: '2px 2px 4px rgba(0,0,0,0.8)',
              maxWidth: '85%',
              textAlign: 'center'
            }}
          />
        </AbsoluteFill>
      )}
      
      {/* Main Audio */}
      {audioSrc && <Audio src={audioSrc} />}

      {/* Sound Effects */}
      {sfx.map((sound, i) => (
        <Sequence key={`sfx-${i}`} from={sound.frame} durationInFrames={sound.duration || 30}>
          <Audio src={sound.src} volume={sound.volume || 0.7} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

// ============== ROOT COMPOSITION ==============

import { Composition } from 'remotion';

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="KidsFactsVideo"
        component={EffectsLayer}
        durationInFrames={1800} // 60 seconds at 30fps
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          videoSrc: '',
          audioSrc: '',
          effects: [],
          captions: [],
          captionStyle: {},
          sfx: []
        }}
      />
    </>
  );
};
