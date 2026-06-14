import { globalScene } from "#app/global-scene";
import { getPokemonNameWithAffix } from "#app/messages";
import { SummonPhase } from "#phases/summon-phase";
import i18next from "i18next";

export class SummonMissingPhase extends SummonPhase {
  public readonly phaseName = "SummonMissingPhase";
  preSummon(): void {
    // A slot's party member can be fainted/illegal here (e.g. a lead that fainted to end-of-turn
    // damage at the previous battle's boundary). Resolve it the same way the base SummonPhase does
    // so we never send a fainted Pokemon back out (which soft-locks the battle).
    if (this.resolveIllegalSummonTarget()) {
      return;
    }
    globalScene.ui.showText(
      i18next.t("battle:sendOutPokemon", {
        pokemonName: getPokemonNameWithAffix(this.getPokemon()),
      }),
    );
    globalScene.time.delayedCall(250, () => this.summon());
  }
}
