"""Python Script Template."""

from exps.post_process import parse_results, print_df

base_dir = "runs/Reacher3Denv"
for agent in ["MPC", "MBMPO"]:
    df = parse_results(base_dir, agent)
    print(agent)
    print_df(df)

    df.to_pickle(f"./{agent}.pk")
