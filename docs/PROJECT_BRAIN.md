# Project Brain

Goal: one brain, many limbs. Keep Blender moving now; make later ports cheap.

- Hosts (later): Blender, Unity, Unreal, SolidWorks, Others
- Providers (now): Meshy; (later): others
- Core (later): minimal interfaces + registry to route jobs

Decisions that make future ports easy:
1) Keep providers free of Blender imports (only mesh import lives in Blender code).
2) Keep top-level entrypoint minimal & lazy; log errors instead of raising.
3) Write tiny “contracts” for Provider/Host in core/ when ready; don’t use until stable.

Roadmap:
- v0.5 (now): Blender-only + Meshy; docs + simple stubs for future core.
- v0.7: Introduce core/contracts.py and a small registry.py, not wired yet.
- v1.0: Move provider to providers/meshy; add HostAdapter for Blender; wire registry.
