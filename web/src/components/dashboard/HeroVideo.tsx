export function HeroVideo() {
  return (
    <div className="w-full h-full bg-black">
      <video
        autoPlay
        muted
        loop
        playsInline
        className="w-full h-full object-contain opacity-80 mix-blend-screen"
      >
        <source src="/scanning-video.mp4" type="video/mp4" />
      </video>
      {/* Vignette overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-panel-dark/80 via-transparent to-transparent pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-r from-background/90 via-background/20 to-transparent pointer-events-none" />
    </div>
  );
}
