# Unfinished: doorway duck, go-around, person-lock fix (NOT for flights)

Draft kept so the work is not lost. It applies on ReachGlass 2302928 after `../reachglass_flysteer.patch`, in place of
`../reachglass_follow_avoid.patch` (it contains that patch's changes plus the draft). Do not use it for flights.

| Part | State in the sim (3 seeds per row, see FOLLOW_AVOID_WIP.md) |
|---|---|
| Duck under a door header | Does not work: fires in 1 of 6 doorway runs, 0 get through. Off by default. |
| Go around a lamp | Partial: passed in 3 of 6 runs, no collisions, closest approach 0.30 to 0.37 m (0.18 m with avoidance off). |
| Go around looming-only obstacles (box, bookcase) | Mostly still stalls: the obstacle's extent is unknown and walls are not mapped. |
| Person lock | Real bug in ReachGlass: the lock picks the LARGEST person box, so from 2 m up a bystander can win over the wearer (only the wearer's head is visible). The draft locks the nearest person by range and remembers bystanders' positions. Worth fixing on its own (perception owner). |

Next steps if resumed: pass through once ducked (their range estimate says "too close" while ducked below head
height), and a wall-aware lateral limit for the go-around; each needs a focused round with small seed batches.
