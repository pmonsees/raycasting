# raycasting
I went with a different approach to this model, which yielded similar results.

Instead of doing Phong-lighting directly, I propagated rays more traditionally.
The reason behind this was because I was not entirely sure how to determine the direction to the light source?
But the results that I got were similar to phong reflection, if a lot more pixelated.

A lot of the implementation for this algorithm came from [this video](https://www.youtube.com/watch?v=Qz0KTGYJtUk),
which I highly recommend passing on to the next generation of Computer Graphic Algorithm students.

Some choice results from my algorithm:

Two orbs with only diffuse bouncing:

![Two orbs with only diffuse lighting](ray_traced_images/two_orbs_only_diffuse.png)

Two orbs with only specular bouncing:

![Two orbs with only specular lighting](ray_traced_images/two_orbs_perfectly_reflective.png)

Two orbs, one that is reflective and one that isn't:

![Two orbs, one reflective, one not](ray_traced_images/two_orbs_5_bounces_1_mirror.png)

Two orbs, one reflective, on a mirror plane

![Two orbs, one reflective, on a mirror plane](ray_traced_images/two_orbs_5_bounces_1_shiny_orb_mirror_plane.png)

Three mirrors all touching

![Two mirror orbs on a mirror](ray_traced_images/5_bounces_3_touching_half_specular_power.png)

Thank you so much, Professor Ribeiro, for this class. I found it deeply interesting, and enjoyed the work I did for this class.
I have learned so much about topics I wasn't aware existed before, and was able to explore them first hand.
This class was amazing, and I can tell that the future classes will enjoy an even more refined version.

All the best, Autumn Monsees