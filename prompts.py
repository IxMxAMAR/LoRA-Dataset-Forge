"""Prompt library for LoRA-Dataset-Forge.

Combinatorial axes, deterministic per-character shuffle. At count=N we produce
N unique specs drawn from the Cartesian product of the axis pools. Capacity is
well over 1B combinations with outfit variation enabled.
"""

import hashlib
import random

# ---------------------------------------------------------------------------
# System prompt — identity lock + quality + anatomy + anti-AI-look
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are generating a training dataset image for a character LoRA. The SAME person appears in every image across this entire dataset.

IDENTITY LOCK (non-negotiable):
- Rigorously preserve the facial identity shown in Reference Image 1 (face reference): exact bone structure, eye shape and eye color, nose shape, lips, jawline, brow shape, skin tone, freckles or marks.
- Preserve the hair color, hair texture, hair length, and hairline shown in the references.
- Preserve the body proportions, build, and implied height from Reference Image 2 (body reference).
- The character is a consistent, real-looking human. Do not stylize, idealize, or age-shift.

VARIATION (what changes per image):
- Pose, expression, framing, camera angle, lighting, environment, weather, props, and (when specified) outfit.

COHERENCE RULE:
- If the specified outfit and environment would be contextually incoherent (e.g. evening gown in a gym, winter coat on a beach), adapt the environment to suit the outfit rather than producing a nonsensical scene. Identity preservation takes precedence over strict environment fidelity.

ANATOMY & QUALITY (non-negotiable):
- Sharp focus on the face and eyes. Both eyes correctly rendered with accurate catchlights and pupils.
- Anatomically correct hands: exactly five fingers per hand, no warping, no fused digits, no extra fingers, no missing fingers.
- Correct limb count and natural proportions. No floating, extra, or missing limbs.
- Natural skin texture with visible pores and subtle imperfections. No plastic skin. No airbrushed over-smoothing. No uncanny symmetry.
- Natural photographic quality. Raw, un-retouched look is preferred over glossy or AI-polished.
- Lighting physics must be coherent with the described environment. Consistent shadow direction, realistic falloff, believable ambient bounce.
- No text, logos, watermarks, borders, captions, split-frames, collages, or UI elements.
- Single person only unless explicitly specified.
"""

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

FRAMINGS = [
    "tight close-up portrait",
    "extreme close-up beauty shot",
    "close-up portrait",
    "bust shot (head and shoulders)",
    "waist-up shot",
    "half-body shot",
    "three-quarter body shot (head to mid-thigh)",
    "full body shot, head to feet",
    "full body wide shot with environment visible",
    "medium shot from the hips up",
]

ANGLES = [
    # Front + three-quarter angles dominate (5/10) — these are what binds
    # facial identity in a LoRA. Both eyes visible, clear bone structure.
    "front-facing, eye level",
    "three-quarter left turn",
    "three-quarter right turn",
    "slight three-quarter left, looking just past the camera",
    "slight three-quarter right, looking just past the camera",
    # Over-shoulder variants still show most of the face.
    "looking over the left shoulder toward the camera",
    "looking over the right shoulder toward the camera",
    # Low / high angle keep both eyes in frame.
    "slight low angle, camera below eye level looking up",
    "slight high angle, camera above looking down",
    # One back-with-turn — shows hair + partial face.
    "back view with the head turned toward the camera",
    # NOTE: dead 90° side profiles deliberately removed. They were ~20% of
    # the old pool and hurt LoRA identity training (one eye hidden, weak
    # facial signature). If you specifically want profile shots, add them
    # back here — but for a character LoRA, 0-10% profile coverage is the
    # industry norm.
]

EXPRESSIONS = [
    "soft natural smile with eyes slightly crinkled",
    "calm neutral gaze directly at the camera",
    "quiet genuine laugh caught mid-moment",
    "subtle knowing smirk",
    "lips slightly parted, thoughtful",
    "serious editorial expression, jaw relaxed",
    "eyes closed, serene and composed",
    "confident, chin slightly raised",
    "looking away pensively, eyes soft",
    "warm closed-lip smile",
    "surprised, brows raised slightly",
    "gentle expression with eyes looking just past the camera",
    "amused half-smile, head tilted",
    "relaxed and natural, no performed expression",
]

LIGHTINGS = [
    "soft natural window light from the left",
    "soft natural window light from the right",
    "bright overcast daylight, diffused and even",
    "warm golden hour sunlight from a low angle",
    "late afternoon sun creating long soft shadows",
    "crisp studio softbox key light with gentle fill",
    "dramatic cinematic rim light outlining the hair and shoulders",
    "warm tungsten indoor lamp light",
    "bright midday sun with a reflector fill",
    "moody low-key lighting with deep shadow falloff",
    "soft ring light, flat and even across the face",
    "early morning soft golden light",
    "blue-hour twilight ambient light",
    "practical lights at night — warm bulbs nearby",
    "harsh directional midday sun creating high-contrast shadows",
    "soft beauty-dish light from slightly above",
    "window light filtered through sheer curtains",
    "backlit silhouette edge with soft fill from the front",
]

# Scenes couple pose + environment + outdoor flag so weather applies coherently.
SCENES = [
    # --- Studio / clean neutral (6) ---
    {"pose": "standing still with shoulders relaxed",
     "environment": "seamless neutral grey photography backdrop", "outdoor": False},
    {"pose": "standing with arms loosely crossed",
     "environment": "clean white cyclorama studio", "outdoor": False},
    {"pose": "standing with one hand resting on the hip",
     "environment": "deep charcoal grey studio backdrop", "outdoor": False},
    {"pose": "seated on a simple wooden chair, legs crossed",
     "environment": "empty photography studio with a single chair and hardwood floor", "outdoor": False},
    {"pose": "standing with feet shoulder-width apart, arms relaxed",
     "environment": "moody dim warehouse interior with a single beam of window light", "outdoor": False},
    {"pose": "hands on hips in a confident stance",
     "environment": "seamless black studio backdrop", "outdoor": False},

    # --- Urban exterior (10) ---
    {"pose": "standing with both hands in pockets",
     "environment": "industrial concrete wall, out of focus", "outdoor": True},
    {"pose": "leaning back against a wall, one foot up on the wall",
     "environment": "sunlit white plaster exterior wall", "outdoor": True},
    {"pose": "mid-stride walking casually forward",
     "environment": "tree-lined city sidewalk with shallow depth of field", "outdoor": True},
    {"pose": "crossing at a crosswalk",
     "environment": "busy downtown intersection with blurred cars", "outdoor": True},
    {"pose": "leaning on a railing looking out",
     "environment": "pedestrian bridge with city skyline behind", "outdoor": True},
    {"pose": "sitting on a low stone wall",
     "environment": "european cobblestone plaza with old buildings", "outdoor": True},
    {"pose": "one hand brushing hair back from the face",
     "environment": "exposed brick wall in an alley, softly out of focus", "outdoor": True},
    {"pose": "mid-motion turning toward the camera",
     "environment": "empty cobblestone alleyway in the old quarter", "outdoor": True},
    {"pose": "standing by a parked scooter",
     "environment": "narrow european side street with storefronts", "outdoor": True},
    {"pose": "descending a short set of stone steps",
     "environment": "steep hillside stairway in a coastal town", "outdoor": True},

    # --- Nature exterior (10) ---
    {"pose": "walking along the edge of the water",
     "environment": "sandy beach with ocean and horizon softly blurred", "outdoor": True},
    {"pose": "sitting on wooden steps, elbows on knees",
     "environment": "exterior wooden porch of a quiet house at the edge of woods", "outdoor": True},
    {"pose": "arms crossed loosely at the chest",
     "environment": "rooftop at magic hour with the city skyline softly blurred", "outdoor": True},
    {"pose": "seated on a park bench",
     "environment": "quiet park path with autumn leaves on the ground", "outdoor": True},
    {"pose": "walking down a forest path",
     "environment": "dense forest trail with dappled light", "outdoor": True},
    {"pose": "standing at a wooden railing",
     "environment": "lakeside pier extending over still water", "outdoor": True},
    {"pose": "leaning on a garden gate",
     "environment": "overgrown country garden with wildflowers", "outdoor": True},
    {"pose": "seated on a large rock",
     "environment": "mountain overlook with valley below", "outdoor": True},
    {"pose": "standing among tall grass",
     "environment": "open meadow with distant tree line", "outdoor": True},
    {"pose": "crossing a small wooden footbridge",
     "environment": "botanical garden with koi pond beneath", "outdoor": True},

    # --- Home / private interior (12) ---
    {"pose": "leaning shoulder against a wooden doorframe",
     "environment": "cozy home interior with wooden door and hallway beyond", "outdoor": False},
    {"pose": "seated on the floor, legs folded to one side",
     "environment": "sunlit wooden floor of a bright loft apartment", "outdoor": False},
    {"pose": "seated on a bed, legs tucked beneath",
     "environment": "simple bedroom with white linen sheets and a single pillow", "outdoor": False},
    {"pose": "seated on a couch, leaning slightly forward",
     "environment": "modern living room with plants in soft bokeh behind", "outdoor": False},
    {"pose": "standing at a kitchen counter",
     "environment": "bright modern kitchen with subway tile and hanging copper pans", "outdoor": False},
    {"pose": "body angled away with head turned back toward the camera",
     "environment": "sunlit bedroom with linen curtains softly glowing", "outdoor": False},
    {"pose": "holding a jacket draped over one arm",
     "environment": "bright minimalist foyer with a large window and a console table", "outdoor": False},
    {"pose": "seated cross-legged on a rug",
     "environment": "warm reading nook with a stack of books and a floor lamp", "outdoor": False},
    {"pose": "standing in front of a tall mirror",
     "environment": "white-tiled bathroom with soft morning light", "outdoor": False},
    {"pose": "curled up in an armchair",
     "environment": "quiet home library with floor-to-ceiling bookshelves", "outdoor": False},
    {"pose": "seated on a kitchen stool at an island",
     "environment": "warm farmhouse kitchen with wooden beams overhead", "outdoor": False},
    {"pose": "leaning on a balcony doorframe looking out",
     "environment": "open balcony door of a city apartment, sheer curtains moving", "outdoor": False},

    # --- Work / public interior (8) ---
    {"pose": "seated at a small table holding a ceramic mug",
     "environment": "quiet café with blurred espresso machine in the background", "outdoor": False},
    {"pose": "standing in an aisle with a book half-open",
     "environment": "independent bookstore with warm wooden shelves", "outdoor": False},
    {"pose": "seated at a long library table",
     "environment": "reading room of an old library with green-shade desk lamps", "outdoor": False},
    {"pose": "walking through a doorway",
     "environment": "minimalist art gallery with white walls and framed works", "outdoor": False},
    {"pose": "seated in a booth by the window",
     "environment": "diner interior with checkered floor and red vinyl seats", "outdoor": False},
    {"pose": "seated on a stool at a counter",
     "environment": "low-lit bar with amber bottles and pendant lights behind", "outdoor": False},
    {"pose": "standing by a grand piano",
     "environment": "intimate concert hall with warm wood paneling", "outdoor": False},
    {"pose": "seated at a mid-century desk",
     "environment": "modern home office with large window and clean surfaces", "outdoor": False},

    # --- Atmospheric / specialty (6) ---
    {"pose": "walking through a passageway",
     "environment": "empty subway station platform with tile walls", "outdoor": False},
    {"pose": "seated in a subway car",
     "environment": "empty subway car with motion-blurred window behind", "outdoor": False},
    {"pose": "standing under an archway",
     "environment": "empty parking garage with clean concrete floor and pillars", "outdoor": False},
    {"pose": "seated on a wide hotel lobby sofa",
     "environment": "grand hotel lobby with marble floor and chandelier", "outdoor": False},
    {"pose": "seated on a wooden bench in a locker-room style space",
     "environment": "dance studio with mirrored wall and bar", "outdoor": False},
    {"pose": "kneeling on a yoga mat",
     "environment": "bright minimalist yoga studio with plants", "outdoor": False},
]

OUTFITS = [
    # Casual (8)
    "a simple fitted white t-shirt and blue jeans",
    "a soft cream cable-knit sweater over dark jeans",
    "a casual oversized hoodie and matching joggers",
    "a fitted grey henley with rolled sleeves and khaki chinos",
    "a vintage band tee tucked into high-waisted jeans",
    "a flannel button-down shirt open over a white tank and jeans",
    "a light denim jacket over a plain tee and black leggings",
    "a cropped beige cardigan over a white camisole and wide jeans",

    # Business / smart (6)
    "a crisp white button-down shirt tucked into tailored black trousers",
    "a fitted blazer over a silk camisole and tailored pants",
    "a chic trench coat over a turtleneck and a pencil skirt",
    "a tailored charcoal suit with a soft-collar blouse",
    "a navy blazer over a cream blouse and cropped trousers",
    "a sleek pencil skirt with a tucked-in silk shirt",

    # Dress / elegant (6)
    "an elegant black cocktail dress",
    "a flowing summer sundress with a soft floral print",
    "a cozy oversized cardigan layered over a simple slip dress",
    "a deep emerald satin slip dress with thin straps",
    "a white linen midi dress with a wrap waist",
    "a structured little black dress with a mock neck",

    # Relaxed / loungewear (4)
    "a matching ribbed lounge set — tank top and wide-leg pants",
    "an oversized men's-cut shirt worn loose over shorts",
    "a soft sweatshirt tucked into sweatpants with white sneakers",
    "a chunky knit sweater over bike shorts",

    # Outdoor / layered (4)
    "a leather jacket over a plain tee and jeans",
    "a long wool coat over a turtleneck and slim trousers",
    "a puffer vest over a long-sleeve top and dark jeans",
    "a trench coat belted over a blouse and slim jeans",

    # Activewear / specialty (4)
    "matching activewear — a sports bra and high-waisted leggings",
    "a yoga tank and high-waisted leggings",
    "a flowing bohemian maxi dress with delicate embroidery",
    "a linen blouse paired with wide-leg linen pants",
]

# PROPS — handheld / hand-placement cues. None entries control "no prop" probability.
# Ratio: 10 None entries vs 28 props → ~26% of images have no prop.
PROPS = [
    None, None, None, None, None,
    None, None, None, None, None,
    # Drinks / food
    "holding a takeout coffee cup with a lid",
    "holding a ceramic mug with both hands",
    "holding a wine glass loosely by the stem",
    "holding a small paper bakery bag",
    # Reading / writing
    "holding an open hardcover book",
    "holding a paperback novel",
    "carrying a stack of three books",
    "holding a small leather-bound notebook",
    # Tech
    "scrolling a smartphone with one hand",
    "holding a smartphone near the ear",
    "wearing over-ear headphones around the neck",
    "carrying a film camera on a strap",
    # Bags / accessories
    "holding a structured leather handbag",
    "carrying a canvas tote bag over the shoulder",
    "holding sunglasses in one hand",
    "with sunglasses pushed up onto the hair",
    # Nature / soft
    "holding a small bouquet of wildflowers",
    "holding a single cut flower stem",
    "holding a vinyl record sleeve",
    # Self-contact / body language
    "one hand resting at the collar of the outfit",
    "hands wrapped around a hot mug",
    "one hand adjusting a small earring",
    "hands tucked into opposite sleeves",
    "arms hugging a soft knit cardigan closed",
    "hand gently resting on the back of the neck",
    "fingertips brushing the jawline",
]

# Weather — applied both outdoor (directly) and indoor (as "visible through the window / weather outside").
WEATHERS = [
    "clear sunny skies",
    "bright overcast daylight",
    "light drizzle beginning to fall",
    "soft fog settled over the scene",
    "fine snowfall drifting down",
    "dramatic storm clouds gathering",
    "wet pavement after recent rain",
    "hazy late-afternoon warmth",
    "crisp clear cold morning",
    "golden autumn afternoon with leaves in the air",
]

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _seed_for(trigger: str) -> int:
    h = hashlib.sha256(trigger.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def combinatorial_capacity(vary_outfit=False):
    base = (
        len(FRAMINGS) * len(ANGLES) * len(EXPRESSIONS)
        * len(LIGHTINGS) * len(SCENES) * len(PROPS) * len(WEATHERS)
    )
    return base * (len(OUTFITS) if vary_outfit else 1)


def decode_flat(flat_index, vary_outfit=False):
    """Decode a flat combinatorial index into (spec, outfit)."""
    i = flat_index
    outfit_n = len(OUTFITS) if vary_outfit else 1
    oi = i % outfit_n if vary_outfit else 0
    i //= outfit_n
    wi = i % len(WEATHERS); i //= len(WEATHERS)
    pi = i % len(PROPS);    i //= len(PROPS)
    si = i % len(SCENES);   i //= len(SCENES)
    li = i % len(LIGHTINGS); i //= len(LIGHTINGS)
    ei = i % len(EXPRESSIONS); i //= len(EXPRESSIONS)
    ai = i % len(ANGLES); i //= len(ANGLES)
    fi = i % len(FRAMINGS)

    scene = SCENES[si]
    spec = {
        "framing": FRAMINGS[fi],
        "angle": ANGLES[ai],
        "expression": EXPRESSIONS[ei],
        "pose": scene["pose"],
        "environment": scene["environment"],
        "lighting": LIGHTINGS[li],
        "prop": PROPS[pi],
        "weather": WEATHERS[wi],
        "outdoor": scene["outdoor"],
    }
    outfit = OUTFITS[oi] if vary_outfit else None
    return spec, outfit


def plan_indices(count, vary_outfit, trigger, exclude=None):
    """Return up to `count` unique flat indices, deterministic per trigger,
    skipping any indices in `exclude`.

    Uses aggressive oversampling plus an iterative top-up so that large
    `exclude` sets don't cause silent short-delivery.
    """
    exclude = set(exclude or [])
    rng = random.Random(_seed_for(trigger))
    total = combinatorial_capacity(vary_outfit=vary_outfit)
    if count <= 0 or total <= 0:
        return []
    need = min(count, max(total - len(exclude), 0))
    if need <= 0:
        return []

    picked: list[int] = []
    seen = set(exclude)
    # Start with a large initial sample; top up iteratively if still short.
    initial_k = min(need * 20 + 500, total)
    attempts_left = 4  # up to 4 oversample rounds
    while len(picked) < need and attempts_left > 0:
        k = min(initial_k, total)
        try:
            candidates = rng.sample(range(total), k=k)
        except ValueError:
            break
        for c in candidates:
            if c in seen:
                continue
            picked.append(c)
            seen.add(c)
            if len(picked) >= need:
                break
        attempts_left -= 1
        # Grow sample size if we're still short
        initial_k = min(initial_k * 4, total)
    return picked[:need]


def plan_jobs(count, vary_outfit=False, trigger="default", exclude=None):
    """Return `count` (flat_index, spec, outfit) tuples.

    `exclude` is an iterable of flat indices to skip (used when regenerating
    rejected slots so the new specs differ from anything previously produced).
    """
    indices = plan_indices(count, vary_outfit, trigger, exclude=exclude)
    return [(flat, *decode_flat(flat, vary_outfit)) for flat in indices]


# ---------------------------------------------------------------------------
# Prompt + caption rendering
# ---------------------------------------------------------------------------

def build_prompt_text(spec, trigger, outfit=None):
    outfit_line = (
        f"Outfit: {outfit}."
        if outfit else
        "Outfit: keep the same outfit shown in the body reference image."
    )
    prop_line = f"Prop / hand action: {spec['prop']}." if spec.get("prop") else ""
    if spec.get("outdoor"):
        weather_line = f"Weather: {spec['weather']}."
    else:
        weather_line = f"Weather visible through windows: {spec['weather']}."

    lines = [
        f"Framing: {spec['framing']}.",
        f"Camera angle: {spec['angle']}.",
        f"Expression: {spec['expression']}.",
        f"Pose: {spec['pose']}.",
        f"Lighting: {spec['lighting']}.",
        f"Environment: {spec['environment']}.",
        weather_line,
        outfit_line,
    ]
    if prop_line:
        lines.append(prop_line)
    lines.append("")
    lines.append(
        "Generate a single photorealistic image of the same person from the references, "
        "matching the specifications above. The face and body identity must be pixel-consistent "
        "with the reference images. Sharp focus on the face, natural skin texture with visible pores, "
        "anatomically correct hands with five fingers each, no AI-generated gloss."
    )
    return "\n".join(lines)


def build_caption(spec, trigger, outfit=None):
    """Qwen-Image-style caption: single line, trigger first, natural language.

    Kohya/musubi/ai-toolkit convention: the trigger word appears ONCE, at the
    start. Avoids re-binding the trigger to framing descriptors instead of the
    character identity.
    """
    outfit_phrase = f", wearing {outfit}" if outfit else ""
    prop_phrase = f", {spec['prop']}" if spec.get("prop") else ""
    weather_phrase = (
        f", {spec['weather']}"
        if spec.get("outdoor") else
        f", {spec['weather']} visible through the window"
    )
    caption = (
        f"{trigger}, {spec['framing']}{outfit_phrase}, "
        f"{spec['angle']}, {spec['expression']}, {spec['pose']}{prop_phrase}, "
        f"{spec['environment']}{weather_phrase}, {spec['lighting']}. "
        f"photorealistic portrait."
    )
    return " ".join(caption.split())


# ---------------------------------------------------------------------------
# Framing categories — used by per-character distribution control
# ---------------------------------------------------------------------------

FRAMING_CATEGORIES = {
    "close": [
        "tight close-up portrait",
        "extreme close-up beauty shot",
        "close-up portrait",
        "bust shot (head and shoulders)",
    ],
    "mid": [
        "waist-up shot",
        "half-body shot",
        "medium shot from the hips up",
    ],
    "full": [
        "three-quarter body shot (head to mid-thigh)",
        "full body shot, head to feet",
        "full body wide shot with environment visible",
    ],
}

CATEGORY_ORDER = ("close", "mid", "full", "random")


def category_for(framing: str) -> str:
    for cat, fs in FRAMING_CATEGORIES.items():
        if framing in fs:
            return cat
    return "random"


def slot_to_category(slot: int, distribution: dict) -> str:
    """Given a 1-indexed slot number, return which category it belongs to per distribution.

    Returns "random" for any slot outside [1, sum(distribution.values())] — this
    function is best-effort; callers should ensure slot ranges are in-bounds.
    """
    if slot < 1:
        return "random"
    cum = 0
    for cat in CATEGORY_ORDER:
        n = distribution.get(cat, 0)
        if slot <= cum + n:
            return cat
        cum += n
    return "random"


def plan_jobs_distributed(distribution: dict, vary_outfit: bool, trigger: str,
                          exclude=None) -> list:
    """Return (flat, spec, outfit, category) for each slot matching the distribution.

    Stratified by framing within each category — round-robins through the
    category's framings so the user reliably gets coverage of every framing
    kind they asked for, instead of RNG variance producing (e.g.) zero
    "full body head to feet" shots in a 7-slot "full" category request.

    Deterministic per (trigger, category) so regenerating rejected slots
    produces fresh picks that remain consistent across runs.
    """
    exclude = set(exclude or [])
    jobs: list = []
    used = set(exclude)

    outfit_n = len(OUTFITS) if vary_outfit else 1
    # Size of the sub-space below the framing axis. Any flat index with a
    # given framing fi is of the form: fi * axes_below + inner, where
    # inner is in [0, axes_below). This lets us directly construct flat
    # indices for a specific target framing.
    axes_below = (outfit_n * len(WEATHERS) * len(PROPS) * len(SCENES)
                  * len(LIGHTINGS) * len(EXPRESSIONS) * len(ANGLES))

    for cat in CATEGORY_ORDER:
        needed = distribution.get(cat, 0)
        if needed <= 0:
            continue

        framings_to_use = (FRAMING_CATEGORIES.get(cat) if cat != "random" else FRAMINGS)
        if not framings_to_use:
            continue

        rng = random.Random(_seed_for(f"{trigger}__{cat}"))
        # Deterministic framing order, different per trigger
        shuffled = list(framings_to_use)
        rng.shuffle(shuffled)

        for slot_idx in range(needed):
            target_framing = shuffled[slot_idx % len(shuffled)]
            target_fi = FRAMINGS.index(target_framing)
            base = target_fi * axes_below
            # Find a non-excluded inner index. Collision rate is negligible
            # (axes_below is millions, exclude is at most thousands).
            attempts = 0
            while attempts < 1000:
                inner = rng.randrange(axes_below)
                flat = base + inner
                if flat in used:
                    attempts += 1
                    continue
                spec, outfit = decode_flat(flat, vary_outfit)
                jobs.append((flat, spec, outfit, cat))
                used.add(flat)
                break

    return jobs


def smart_aspect(framing: str) -> str:
    """Pick a training-friendly aspect ratio per framing.

    Tuned for tall character subjects:
    - Close-ups → 3:4 (keeps face pixel density high)
    - Mid-shots → 4:5 (slight portrait; square would crop shoulders)
    - All full-body variants → 2:3 (classic portrait, shows height with
      comfortable context on the sides)
    9:16 was dropped after user testing — too narrow for most character
    LoRA training use cases. If you want it back for specific framings,
    edit this mapping.
    """
    f = framing.lower()
    if "extreme close" in f or "tight close" in f or "close-up" in f or "bust" in f:
        return "3:4"
    if "waist-up" in f or "half-body" in f or "medium shot" in f:
        return "4:5"
    if "three-quarter body" in f or "full body" in f:
        return "2:3"
    return "3:4"
