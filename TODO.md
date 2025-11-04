# TODO - Fix RL Simulation Mode

## Current Status
- Current `app.py` only contains `run_rl_simulation` method without class structure
- Missing main function and argument parsing for `--mode sim_rl_100 --additional-days 100`
- Need to integrate RL agent into TradingSystem class

## Tasks
- [ ] Create complete `app.py` with TradingSystem class
- [ ] Add RL agent initialization in `__init__`
- [ ] Add `sim_rl_100` mode to argument parser
- [ ] Add `--additional-days` argument support
- [ ] Update main function to handle RL simulation mode
- [ ] Test the command execution

## Next Steps
- Run: `python src/app.py --mode sim_rl_100 --additional-days 100`
- Verify RL simulation runs for 100 days
- Check for any errors in data loading or feature processing
