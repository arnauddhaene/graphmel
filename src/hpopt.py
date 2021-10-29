import click

from ray import tune

from run import run


@click.command()
@click.option('--model', default='GAT',
              type=click.Choice(['GNN', 'GAT', 'GIN', 'DiffPool'], case_sensitive=False),
              help="Model architecture choice.")
@click.option('--connectivity', default='wasserstein',
              type=click.Choice(['fully', 'organ', 'wasserstein'], case_sensitive=False),
              help="Graph connectivity choice.")
@click.option('--epochs', default=50,
              help="Number of training epochs.")
@click.option('--test-size', default=0.2,
              help="Test set size in ratio.")
@click.option('--seed', default=27,
              help="Random seed.")
@click.option('--cv', default=5,
              help="Cross-validation splits.")
@click.option('--verbose', default=1, type=int,
              help="Print out info for debugging purposes.")
@click.pass_context
def tune_hyperparams(ctx: click.Context, model, connectivity, epochs,
                     test_size, seed, cv, verbose) -> None:
    
    analysis = tune.run(
        invoke_run,
        config=dict(
            context=ctx, verbose=verbose,
            model=model,
            connectivity=connectivity, epochs=epochs,
            test_size=test_size, seed=seed, cv=cv,
            lr=tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4]),
            decay=tune.grid_search([1e-2, 1e-3]),
            hidden_dim=tune.choice([16, 32, 64, 128]),
            batch_size=tune.choice([4, 8, 16])
        ))
    
    result = analysis.get_best_config(metric='objective', mode='max')
    
    print(f"Best configuration: {result}")

    
def invoke_run(config):
    
    ctx, model, connectivity, epochs, test_size, seed, cv, verbose = \
        config['context'], config['model'], config['connectivity'], config['epochs'], \
        config['test_size'], config['seed'], config['cv'], config['verbose']
    
    # Extract hyperparameters
    lr, decay, hidden_dim, batch_size = \
        config['lr'], config['decay'], config['hidden_dim'], config['batch_size']
    
    metric = ctx.invoke(run, model=model, connectivity=connectivity, epochs=epochs,
                        lr=lr, decay=decay, hidden_dim=hidden_dim, batch_size=batch_size,
                        test_size=test_size, seed=seed, cv=cv,
                        experiment_name='Default', verbose=verbose)
    
    tune.report(objective=metric)


if __name__ == '__main__':
    tune_hyperparams()
