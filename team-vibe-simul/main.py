from scipy.optimize import linprog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import argparse
import time

console = Console()

# Research-based productivity profiles
PROFILES = {
    'conservative': {
        'name': 'Conservative (Research Minimum)',
        'p': 5,     # NLOC/hour - based on open source study
        'r': 200,   # NLOC/hour - 40:1 ratio (research minimum)
        'description': 'Based on open-source project measurements'
    },
    'moderate': {
        'name': 'Moderate (Industry Average)',
        'p': 8,     # NLOC/hour - midpoint of studies
        'r': 250,   # NLOC/hour - 31:1 ratio
        'description': 'Average across multiple research studies'
    },
    'optimistic': {
        'name': 'Optimistic (Commercial Teams)',
        'p': 10,    # NLOC/hour - commercial team data
        'r': 300,   # NLOC/hour - 30:1 ratio
        'description': 'Based on high-performing commercial teams'
    }
}

def solve_basic_dev_team_optimization(n, p, r, H):
    """
    Solve the basic developer team optimization problem (no vibe coding)

    Based on research:
    - Production rates: Studies show 5-10 NLOC/hour for actual development
      (ResearchGate study avg: 7.8 NLOC/hr; Capers Jones: 2-4 NLOC/hr;
       NDepend commercial: 10 NLOC/hr)
    - Review rates: 150-400 NLOC/hour optimal
      (PSP studies: 150-200; SmartBear/Cisco: 200-400 optimal)

    Args:
        n: number of developers
        p: NLOC per hour each dev can produce
        r: NLOC per hour each dev can review
        H: hours available per developer

    Returns:
        dict with optimal allocation and max NLOC produced
    """

    # Objective: maximize n*p*x (we minimize -n*p*x)
    c = [-n * p, 0]  # coefficients for [x, y]

    # Inequality constraints: A_ub * [x, y] <= b_ub
    A_ub = [
        [1, 1],      # x + y <= H (time constraint)
        [p, -r]      # p*x - r*y <= 0 (review constraint: p*x <= r*y)
    ]
    b_ub = [H, 0]

    # Variable bounds: x >= 0, y >= 0
    bounds = [(0, None), (0, None)]

    # Solve with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Optimizing basic model...", total=None)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        progress.update(task, completed=100)

    if result.success:
        x_opt, y_opt = result.x
        max_nloc = n * p * x_opt

        return {
            'model_type': 'basic',
            'optimal_x': x_opt,
            'optimal_y': y_opt,
            'max_nloc_produced': max_nloc,
            'hours_producing': x_opt,
            'hours_reviewing': y_opt,
            'total_hours_used': x_opt + y_opt
        }
    else:
        return {'error': 'Optimization failed', 'message': result.message}

def solve_optimal_vibe_fraction_optimization(n, p_h, p_v, r_h, r_v, H):
    """
    Solve the optimal vibe fraction developer team optimization problem

    Args:
        n: number of developers
        p_h: NLOC per hour for human coding
        p_v: NLOC per hour for vibe coding
        r_h: NLOC per hour for reviewing human code
        r_v: NLOC per hour for reviewing vibe code
        H: hours available per developer

    Returns:
        dict with optimal allocation, max NLOC produced, and optimal vibe fraction
    """

    # Variables: [x_h, x_v, y_h, y_v]

    # Objective: maximize n * (p_h * x_h + p_v * x_v)
    c = [-n * p_h, -n * p_v, 0, 0]  # coefficients for [x_h, x_v, y_h, y_v]

    # Inequality constraints
    A_ub = [
        [1, 1, 1, 1],        # x_h + x_v + y_h + y_v <= H (time constraint)
        [p_h, 0, -r_h, 0],   # p_h * x_h <= r_h * y_h (human code review)
        [0, p_v, 0, -r_v]    # p_v * x_v <= r_v * y_v (vibe code review)
    ]
    b_ub = [H, 0, 0]

    # Variable bounds: all >= 0
    bounds = [(0, None), (0, None), (0, None), (0, None)]

    # Solve with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Finding optimal vibe fraction...", total=None)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        progress.update(task, completed=100)

    if result.success:
        x_h_opt, x_v_opt, y_h_opt, y_v_opt = result.x

        nloc_human = n * p_h * x_h_opt
        nloc_vibe = n * p_v * x_v_opt
        max_nloc = nloc_human + nloc_vibe

        # Calculate optimal vibe fraction
        if max_nloc > 0:
            vibe_fraction_optimal = nloc_vibe / max_nloc
        else:
            vibe_fraction_optimal = 0.0

        return {
            'model_type': 'optimal_vibe',
            'optimal_x_h': x_h_opt,
            'optimal_x_v': x_v_opt,
            'optimal_y_h': y_h_opt,
            'optimal_y_v': y_v_opt,
            'max_nloc_produced': max_nloc,
            'nloc_human': nloc_human,
            'nloc_vibe': nloc_vibe,
            'vibe_fraction_optimal': vibe_fraction_optimal,
            'total_hours_used': x_h_opt + x_v_opt + y_h_opt + y_v_opt,
            'hours_human_coding': x_h_opt,
            'hours_vibe_coding': x_v_opt,
            'hours_reviewing_human': y_h_opt,
            'hours_reviewing_vibe': y_v_opt
        }
    else:
        return {'error': 'Optimization failed', 'message': result.message}

def solve_vibe_coded_dev_team_optimization(n, p_h, p_v, r_h, r_v, alpha, H):
    """
    Solve the vibe-coded developer team optimization problem

    Args:
        n: number of developers
        p_h: NLOC per hour for human coding
        p_v: NLOC per hour for vibe coding
        r_h: NLOC per hour for reviewing human code
        r_v: NLOC per hour for reviewing vibe code
        alpha: fraction of total NLOC that should be vibe coded
        H: hours available per developer

    Returns:
        dict with optimal allocation and max NLOC produced
    """

    if alpha >= 1 or alpha < 0:
        return {'error': 'Alpha must be between 0 and 1'}

    # Variables: [x_h, y_h, y_v] (x_v is determined by vibe fraction constraint)
    # x_v = (alpha * p_h * x_h) / (p_v * (1 - alpha))

    # Objective: maximize n * (p_h * x_h + p_v * x_v)
    # = n * (p_h * x_h + p_v * (alpha * p_h * x_h) / (p_v * (1 - alpha)))
    # = n * p_h * x_h * (1 + alpha / (1 - alpha))
    vibe_multiplier = 1 + alpha / (1 - alpha)
    c = [-n * p_h * vibe_multiplier, 0, 0]  # coefficients for [x_h, y_h, y_v]

    # Inequality constraints
    # 1. Time constraint: x_h + x_v + y_h + y_v <= H
    #    x_h + (alpha * p_h * x_h) / (p_v * (1 - alpha)) + y_h + y_v <= H
    #    x_h * (1 + (alpha * p_h) / (p_v * (1 - alpha))) + y_h + y_v <= H
    time_coeff = 1 + (alpha * p_h) / (p_v * (1 - alpha))

    # 2. Human code review constraint: p_h * x_h <= r_h * y_h
    # 3. Vibe code review constraint: p_v * x_v <= r_v * y_v
    #    p_v * (alpha * p_h * x_h) / (p_v * (1 - alpha)) <= r_v * y_v
    #    (alpha * p_h * x_h) / (1 - alpha) <= r_v * y_v
    vibe_review_coeff = (alpha * p_h) / (1 - alpha)

    A_ub = [
        [time_coeff, 1, 1],           # time constraint
        [p_h, -r_h, 0],               # human code review constraint
        [vibe_review_coeff, 0, -r_v]  # vibe code review constraint
    ]
    b_ub = [H, 0, 0]

    # Variable bounds: x_h, y_h, y_v >= 0
    bounds = [(0, None), (0, None), (0, None)]

    # Solve with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Optimizing vibe-coded model...", total=None)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        progress.update(task, completed=100)

    if result.success:
        x_h_opt, y_h_opt, y_v_opt = result.x
        x_v_opt = (alpha * p_h * x_h_opt) / (p_v * (1 - alpha))

        nloc_human = n * p_h * x_h_opt
        nloc_vibe = n * p_v * x_v_opt
        max_nloc = nloc_human + nloc_vibe

        return {
            'model_type': 'vibe_coded',
            'optimal_x_h': x_h_opt,
            'optimal_x_v': x_v_opt,
            'optimal_y_h': y_h_opt,
            'optimal_y_v': y_v_opt,
            'max_nloc_produced': max_nloc,
            'nloc_human': nloc_human,
            'nloc_vibe': nloc_vibe,
            'vibe_fraction_achieved': nloc_vibe / max_nloc if max_nloc > 0 else 0,
            'total_hours_used': x_h_opt + x_v_opt + y_h_opt + y_v_opt,
            'hours_human_coding': x_h_opt,
            'hours_vibe_coding': x_v_opt,
            'hours_reviewing_human': y_h_opt,
            'hours_reviewing_vibe': y_v_opt
        }
    else:
        return {'error': 'Optimization failed', 'message': result.message}

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Developer Team Optimization Simulator')
    parser.add_argument('--mode', '-m',
                       choices=['conservative', 'moderate', 'optimistic'],
                       default='conservative',
                       help='Productivity profile based on research data (default: conservative)')
    parser.add_argument('--devs', '-n', type=int, default=5,
                       help='Number of developers (default: 5)')
    parser.add_argument('--hours', '-H', type=float, default=8,
                       help='Available hours per developer (default: 8)')
    parser.add_argument('--ai-multiplier', '-a', type=float, default=4.0,
                       help='AI coding multiplier for production (default: 4x)')
    parser.add_argument('--vibe-review-factor', '-v', type=float, default=4.0,
                       help='Vibe coding review slowdown factor (default: 4x slower)')
    parser.add_argument('--vibe-fraction', '-f', type=float, default=0.3,
                       help='Target fraction of code that should be vibe-coded (default: 0.3)')
    parser.add_argument('--custom-prod', '-p', type=float,
                       help='Custom production rate (NLOC/hr)')
    parser.add_argument('--custom-review', '-r', type=float,
                       help='Custom review rate (NLOC/hr)')

    args = parser.parse_args()

    # Use custom rates if provided, otherwise use profile
    if args.custom_prod and args.custom_review:
        p_base = args.custom_prod
        r_base = args.custom_review
        profile_name = "Custom"
        profile_desc = f"User-defined rates"
    else:
        profile = PROFILES[args.mode]
        p_base = profile['p']
        r_base = profile['r']
        profile_name = profile['name']
        profile_desc = profile['description']

    n = args.devs
    H = args.hours
    ai_mult = args.ai_multiplier
    vibe_review_factor = args.vibe_review_factor
    alpha = args.vibe_fraction

    console.print(Panel.fit("[bold cyan]DEVELOPER TEAM OPTIMIZATION SIMULATOR[/bold cyan]", padding=(1, 2)))

    # Input parameters table
    input_table = Table(title="Input Parameters", show_header=False, box=None)
    input_table.add_column("Parameter", style="cyan")
    input_table.add_column("Value", style="yellow")
    input_table.add_row("Profile", f"{profile_name}")
    input_table.add_row("", f"[dim]{profile_desc}[/dim]")
    input_table.add_row("Number of developers", f"{n}")
    input_table.add_row("Available hours", f"{H} hrs per dev")
    input_table.add_row("", "")
    input_table.add_row("[bold]Base Rates[/bold]", "")
    input_table.add_row("Human production", f"{p_base} NLOC/hr")
    input_table.add_row("Human review", f"{r_base} NLOC/hr")
    input_table.add_row("Review/Production ratio", f"{r_base/p_base:.1f}:1")
    input_table.add_row("", "")
    input_table.add_row("[bold]Vibe Coding Factors[/bold]", "")
    input_table.add_row("AI multiplier", f"{ai_mult}x production boost")
    input_table.add_row("Review slowdown", f"{vibe_review_factor}x slower")
    input_table.add_row("Target vibe fraction", f"{alpha:.0%}")
    console.print(input_table)
    console.print()

    # Calculate all five models
    basic_result = solve_basic_dev_team_optimization(n=n, p=p_base, r=r_base, H=H)

    # For vibe model: vibe production is AI-boosted, vibe review is slower
    p_vibe = p_base * ai_mult
    r_vibe = r_base / vibe_review_factor

    # Fixed fraction with normal review
    vibe_result = solve_vibe_coded_dev_team_optimization(
        n=n, p_h=p_base, p_v=p_vibe, r_h=r_base, r_v=r_vibe, alpha=alpha, H=H
    )

    # Fixed fraction with perfect AI (no review needed)
    perfect_result = solve_vibe_coded_dev_team_optimization(
        n=n, p_h=p_base, p_v=p_vibe, r_h=r_base, r_v=1e9, alpha=alpha, H=H
    )

    # Optimal fraction with normal review
    optimal_result = solve_optimal_vibe_fraction_optimization(
        n=n, p_h=p_base, p_v=p_vibe, r_h=r_base, r_v=r_vibe, H=H
    )

    # Optimal fraction with perfect AI (no review)
    optimal_perfect_result = solve_optimal_vibe_fraction_optimization(
        n=n, p_h=p_base, p_v=p_vibe, r_h=r_base, r_v=1e9, H=H
    )

    if ('error' not in basic_result and 'error' not in vibe_result and
        'error' not in perfect_result and 'error' not in optimal_result and
        'error' not in optimal_perfect_result):

        # Comparison results table
        results_table = Table(title="[bold green]Optimization Results[/bold green]", show_lines=True)
        results_table.add_column("Model", style="cyan", width=18)
        results_table.add_column("NLOC\nProduced", style="yellow", justify="right", width=10)
        results_table.add_column("NLOC\nReviewed", style="blue", justify="right", width=10)
        results_table.add_column("Gain", style="green", justify="right", width=6)
        results_table.add_column("Vibe\n%", style="magenta", justify="right", width=6)
        results_table.add_column("Prod Hours\n(H/V)", style="dim", justify="right", width=12)
        results_table.add_column("Rev Hours\n(H/V)", style="dim", justify="right", width=12)

        # Basic model
        nloc_reviewed_basic = basic_result['max_nloc_produced']  # All code must be reviewed
        results_table.add_row(
            "Human only",
            f"{basic_result['max_nloc_produced']:.0f}",
            f"{nloc_reviewed_basic:.0f}",
            "1.0x",
            "0%",
            f"{basic_result['hours_producing']:.1f}/0",
            f"{basic_result['hours_reviewing']:.1f}/0"
        )

        # Fixed fraction with review
        vibe_mult = vibe_result['max_nloc_produced'] / basic_result['max_nloc_produced']
        nloc_reviewed_vibe = vibe_result['max_nloc_produced']  # All code reviewed
        results_table.add_row(
            f"{alpha:.0%} vibe coding",
            f"{vibe_result['max_nloc_produced']:.0f}",
            f"{nloc_reviewed_vibe:.0f}",
            f"{vibe_mult:.1f}x",
            f"{vibe_result['vibe_fraction_achieved']*100:.0f}%",
            f"{vibe_result['hours_human_coding']:.1f}/{vibe_result['hours_vibe_coding']:.1f}",
            f"{vibe_result['hours_reviewing_human']:.1f}/{vibe_result['hours_reviewing_vibe']:.1f}"
        )
        # Add NLOC breakdown rows only
        results_table.add_row(
            f"  [dim]Human[/dim]",
            f"[dim]{vibe_result['nloc_human']:.0f}[/dim]",
            f"[dim]{vibe_result['nloc_human']:.0f}[/dim]",
            "",
            "",
            "",
            ""
        )
        results_table.add_row(
            f"  [dim]Vibe[/dim]",
            f"[dim]{vibe_result['nloc_vibe']:.0f}[/dim]",
            f"[dim]{vibe_result['nloc_vibe']:.0f}[/dim]",
            "",
            "",
            "",
            ""
        )

        # Fixed fraction perfect
        perfect_mult = perfect_result['max_nloc_produced'] / basic_result['max_nloc_produced']
        nloc_reviewed_perfect = perfect_result['nloc_human']  # Only human code needs review
        results_table.add_row(
            f"{alpha:.0%} vibe (no review)",
            f"{perfect_result['max_nloc_produced']:.0f}",
            f"{nloc_reviewed_perfect:.0f}",
            f"{perfect_mult:.1f}x",
            f"{perfect_result['vibe_fraction_achieved']*100:.0f}%",
            f"{perfect_result['hours_human_coding']:.1f}/{perfect_result['hours_vibe_coding']:.1f}",
            f"{perfect_result['hours_reviewing_human']:.1f}/{perfect_result['hours_reviewing_vibe']:.1f}"
        )
        # Add NLOC breakdown rows only
        results_table.add_row(
            f"  [dim]Human[/dim]",
            f"[dim]{perfect_result['nloc_human']:.0f}[/dim]",
            f"[dim]{perfect_result['nloc_human']:.0f}[/dim]",
            "",
            "",
            "",
            ""
        )
        results_table.add_row(
            f"  [dim]Vibe[/dim]",
            f"[dim]{perfect_result['nloc_vibe']:.0f}[/dim]",
            f"[dim]0[/dim]",  # Vibe code doesn't need review in perfect scenario
            "",
            "",
            "",
            ""
        )

        # Separator
        results_table.add_row("", "", "", "", "", "", "", style="dim")

        # Optimal with review
        optimal_pct = optimal_result['vibe_fraction_optimal']*100
        optimal_mult = optimal_result['max_nloc_produced'] / basic_result['max_nloc_produced']
        nloc_reviewed_optimal = optimal_result['max_nloc_produced']  # All code reviewed
        results_table.add_row(
            f"[green]Optimal ({optimal_pct:.0f}% vibe)[/green]",
            f"[green]{optimal_result['max_nloc_produced']:.0f}[/green]",
            f"[green]{nloc_reviewed_optimal:.0f}[/green]",
            f"[green]{optimal_mult:.1f}x[/green]",
            f"[green]{optimal_pct:.0f}%[/green]",
            f"[green]{optimal_result['hours_human_coding']:.1f}/{optimal_result['hours_vibe_coding']:.1f}[/green]",
            f"[green]{optimal_result['hours_reviewing_human']:.1f}/{optimal_result['hours_reviewing_vibe']:.1f}[/green]"
        )
        # Add breakdown rows
        results_table.add_row(
            f"  [dim]Human[/dim]",
            f"[dim]{optimal_result['nloc_human']:.0f}[/dim]",
            f"[dim]{optimal_result['nloc_human']:.0f}[/dim]",
            "",
            "",
            f"[dim]{optimal_result['hours_human_coding']:.1f}[/dim]",
            f"[dim]{optimal_result['hours_reviewing_human']:.1f}[/dim]"
        )
        results_table.add_row(
            f"  [dim]Vibe[/dim]",
            f"[dim]{optimal_result['nloc_vibe']:.0f}[/dim]",
            f"[dim]{optimal_result['nloc_vibe']:.0f}[/dim]",
            "",
            "",
            f"[dim]{optimal_result['hours_vibe_coding']:.1f}[/dim]",
            f"[dim]{optimal_result['hours_reviewing_vibe']:.1f}[/dim]"
        )

        # Optimal perfect
        optimal_perfect_pct = optimal_perfect_result['vibe_fraction_optimal']*100
        optimal_perfect_mult = optimal_perfect_result['max_nloc_produced'] / basic_result['max_nloc_produced']
        nloc_reviewed_optimal_perfect = optimal_perfect_result.get('nloc_human', 0)  # Only human code needs review
        results_table.add_row(
            f"[bright_magenta]Optimal no review[/bright_magenta]",
            f"[bright_magenta]{optimal_perfect_result['max_nloc_produced']:.0f}[/bright_magenta]",
            f"[bright_magenta]{nloc_reviewed_optimal_perfect:.0f}[/bright_magenta]",
            f"[bright_magenta]{optimal_perfect_mult:.1f}x[/bright_magenta]",
            f"[bright_magenta]{optimal_perfect_pct:.0f}%[/bright_magenta]",
            f"[bright_magenta]{optimal_perfect_result['hours_human_coding']:.1f}/{optimal_perfect_result['hours_vibe_coding']:.1f}[/bright_magenta]",
            f"[bright_magenta]{optimal_perfect_result['hours_reviewing_human']:.1f}/{optimal_perfect_result['hours_reviewing_vibe']:.1f}[/bright_magenta]"
        )
        # Add breakdown rows
        results_table.add_row(
            f"  [dim]Human[/dim]",
            f"[dim]{optimal_perfect_result['nloc_human']:.0f}[/dim]",
            f"[dim]{optimal_perfect_result['nloc_human']:.0f}[/dim]",
            "",
            "",
            f"[dim]{optimal_perfect_result['hours_human_coding']:.1f}[/dim]",
            f"[dim]{optimal_perfect_result['hours_reviewing_human']:.1f}[/dim]"
        )
        results_table.add_row(
            f"  [dim]Vibe[/dim]",
            f"[dim]{optimal_perfect_result['nloc_vibe']:.0f}[/dim]",
            f"[dim]0[/dim]",  # Vibe code doesn't need review in perfect scenario
            "",
            "",
            f"[dim]{optimal_perfect_result['hours_vibe_coding']:.1f}[/dim]",
            f"[dim]{optimal_perfect_result['hours_reviewing_vibe']:.1f}[/dim]"
        )

        console.print(results_table)

    else:
        if 'error' in basic_result:
            console.print(f"[red bold]Basic Model Error:[/red bold] {basic_result['error']}")
        if 'error' in vibe_result:
            console.print(f"[red bold]Vibe Model Error:[/red bold] {vibe_result['error']}")
