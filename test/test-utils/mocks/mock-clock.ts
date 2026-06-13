import Phaser from "phaser";

const Clock = Phaser.Time.Clock;

export class MockClock extends Clock {
  public overrideDelay: number | null = 1;
  private readonly tickInterval: ReturnType<typeof setInterval>;
  constructor(scene) {
    super(scene);
    this.tickInterval = setInterval(() => {
      /*
        To simulate frame update
        eventEmitter.on(SceneEvents.PRE_UPDATE, this.preUpdate, this);
        eventEmitter.on(SceneEvents.UPDATE, this.update, this);
       */
      this.preUpdate(this.systems.game.loop.time, 1);
      this.update(this.systems.game.loop.time, 1);
    }, 1);
  }

  /**
   * Stop the 1ms tick interval. MUST be called when this clock is replaced —
   * otherwise every replacement leaks a permanent 1kHz timer that pins the
   * old clock (and whatever it references) in memory. With one MockClock per
   * episode/test in a shared process, that compounds into real CPU and RSS
   * drift (observed: -22% steps/s and +170MB over 42 in-process episodes).
   */
  destroy(): void {
    clearInterval(this.tickInterval);
  }

  addEvent(config: Phaser.Time.TimerEvent | Phaser.Types.Time.TimerEventConfig): Phaser.Time.TimerEvent {
    const cfg = { ...config, delay: this.overrideDelay ?? config.delay ?? 0 };
    return super.addEvent(cfg);
  }
}
