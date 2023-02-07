def wandb_cluster_mode():
    """
    Get wandb key and turn wandb offline. Requires os imported?
    """
    import os
    key = os.environ.get("WANDB_KEY")
    os.environ['WANDB_API_KEY'] = key 
    os.environ['WANDB_MODE'] = 'offline'

def save_torch_model(model, path, name):
    from pathlib import Path
    import torch
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / name)


