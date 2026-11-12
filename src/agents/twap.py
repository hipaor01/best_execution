"""Referencia determinista de ejecución uniforme."""


def twap_action(env):
    """Compra si falta volumen para alcanzar la pauta al final del paso."""
    target = (env.vol_care * (env.state_pos + 1)) // env.episode_bins
    return int(env.exec_vol < target)
