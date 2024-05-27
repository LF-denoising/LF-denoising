import yaml

def save_yaml(opt, yaml_name):
    with open(yaml_name, 'w') as f:
        data = yaml.dump(vars(opt), f)
