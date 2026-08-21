import sys

from sonora import __version__

IS_WINDOWS = sys.platform == 'win32'

FLAC_CMD = "flac.exe" if IS_WINDOWS else "flac"
METAFLAC_CMD = "metaflac.exe" if IS_WINDOWS else "metaflac"
FFMPEG_CMD = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
SOX_CMD = "sox.exe" if IS_WINDOWS else "sox"
BPM_TAG_CMD = "bpm-tag.exe" if IS_WINDOWS else "bpm-tag"

SUPPORTED_EXTS = frozenset({".flac", ".mp3", ".m4a", ".mp4", ".alac", ".ogg", ".opus", ".wav", ".aiff", ".wma", ".ape", ".wv", ".mpc"})

GENRE_MAP = {
    "Hip-Hop": "Hip-Hop/Rap",
    "Hip Hop": "Hip-Hop/Rap",
    "Rap": "Hip-Hop/Rap",
    "Rnb": "R&B",
    "R&B": "R&B/Soul",
    "Electronic": "Electronic",
    "Dance": "Dance",
    "House": "House",
    "Pop/Rock": "Pop",
    "Drum And Bass": "Drum & Bass",
    "Synthpop": "Synth-pop",
    "Alternative Rock": "Alternative",
    "Indie Rock": "Indie",
}

BROAD_GENRE_KEYWORDS = frozenset({
    "hip hop", "rap", "trap", "r&b", "soul", "pop", "rock", "metal", "jazz", "blues", 
    "electronic", "dance", "house", "techno", "indie", "alt", "folk", "country", 
    "regae", "reggae", "punk", "funk", "disco", "classical", "ambient", "lo-fi",
    "drill", "grime", "dubstep", "electro", "synth", "wave", "latin", "reggaeton",
    "garage", "dancehall", "ska", "ska-punk", "hardcore", "grunge", "progressive",
    "psychedelic", "experimental", "instrumental", "vocal", "acoustic", "orchestral"
})

GENRE_BLACKLIST = frozenset({
    "Billboard", "Hot 100", "Top 40", "Amazon", "Itunes",
    "Unknown", "Release", "Music", "Digital", "Various",
    "Wes", "Christoph Andersson", "Produced By", "Written By",
    "Mixed By", "Mastered By", "Engineer", "Composer"
})

SUSPICIOUS_FEATS = frozenset({
    "me", "this", "that", "us", "him", "her", "them", "those", "you", "i", "it", 
    "to", "at", "by", "for", "with", "everything", "zori", "tine", "zilei", 
    "noapte", "seara", "zi", "cu", "si", "și", "feat", "tine până în zori", 
    "până în zori", "zori de zi", "in", "în", "de", "la", "pe", "din", "al",
    "viata", "viața", "lumea", "inima", "sufletul", "cerul", "focul", "apa", 
    "dorul", "gandul", "gândul", "sine", "noi", "morții", "banii", "bani",
    "drag", "călin", "nori", "flori", "gheață", "mine", "voi", "ei", "ele",
    "ea", "el", "mea", "meu", "lor", "tot", "toata", "toată", "toti", "toți",
    "toate", "nimic", "ceva", "cineva", "oricine", "orice", "vreun", "vreo",
    "the devil", "demons", "final fantasy", "please!", "beach", "umbra", "ta", 
    "tău", "lui", "noastră", "vostru", "două", "trei", "patru", "cinci", "șase", "șapte", "opt", "nouă", "zece",
    "sută", "mie", "ani", "timp", "gând", "gânduri", "unde", "care", "cine",
    "ce", "când", "cum", "de ce", "pentru", "fără", "peste", "prin", "spre",
    "după", "lângă", "între", "sub", "dinspre", "înspre", "până", "dintre"
})
# Alias Mappings (Real Names to Stage Names)
ARTIST_ALIASES: dict[str, str] = {
    # M.G.L.
    "matasaru leonard george": "M.G.L.",
    "matasaru george leonard": "M.G.L.",
    "mătăsaru leonard george": "M.G.L.",
    "mătăsaru george leonard": "M.G.L.",
    "mătăsaru george-leonard": "M.G.L.",
    "m.g.l": "M.G.L.",
    "mgl": "M.G.L.",
    
    # Nane
    "stefan avram cherescu": "Nane",
    "ștefan avram cherescu": "Nane",
    "stefan cherescu": "Nane",
    "ștefan cherescu": "Nane",
    "nane": "Nane",
    
    # Killa Fonic
    "ionut raducanu": "Killa Fonic",
    "ionuț răducanu": "Killa Fonic",
    "ionut rapciug": "Killa Fonic",
    "ionuț răpciug": "Killa Fonic",
    "killa fonic": "Killa Fonic",
    
    # Ian & Azteca
    "anghel georgian bogdan": "Ian",
    "bogdan georgian anghel": "Ian",
    "ian": "Ian",
    "andrew edward nedelcu": "Azteca",
    "andrew-edward nedelcu": "Azteca",
    
    # Amuly
    "hameed amil": "Amuly",
    "alexandru mincu": "Amuly",
    
    # Bvcovia & Rava
    "raduly ioan marian": "Bvcovia",
    "ioan marian raduly": "Bvcovia",
    "ravanelli florin oita": "Rava",
    "ravanelli florin oiță": "Rava",
    
    # Noua Unspe
    "ghinea alexandru daniel": "Noua Unspe",
    
    # Satra B.E.N.Z Members
    "darius vlad cretan": "Nosfe",
    "darius vlad crețan": "Nosfe",
    "bogdan david ionita": "Keed",
    "bogdan david ioniță": "Keed",
    "catalin guta": "Super ED",
    "cătălin guță": "Super ED",
    
    # Hip-Hop Legends (CTC, etc)
    "vlad munteanu": "DOC",
    "vlad-costin munteanu": "DOC",
    "razvan eremia": "Deliric",
    "răzvan eremia": "Deliric",
    "deliric": "Deliric",
    "marius stelian craciun": "Cedry2k",
    "marius stelian crăciun": "Cedry2k",
    "mihai adamescu": "Chimie",
    "dragos tudorache": "Dragonu'",
    "dragoș tudorache": "Dragonu'",
    "b.u.g. mafia": "B.U.G. Mafia",
    "bug mafia": "B.U.G. Mafia",
    
    # Pop / Mainstream
    "stefan mihalache": "Connect-R",
    "ștefan mihalache": "Connect-R",
    "laurențiu mocanu": "Guess Who",
    "laurentiu mocanu": "Guess Who",
    "gabriel mihai istrate": "Shift",
    "andrei mihai maria": "Smiley",
    "andrei tiberiu maria": "Smiley",
    "elena alexandra apostoleanu": "Inna",
    "alin emil ghita": "El Nino",
    "alin emil ghiță": "El Nino",
    "maria alexandra florea": "Holy Molly",
    "adriana livia opris": "Olivia Addams",
    "adriana livia opriș": "Olivia Addams",
}


PROTECTED_ARTISTS = frozenset({
    "Play & Win", "Play&Win", "Rauf & Faik", "Rauf&Faik", "Simon & Garfunkel", "Earth, Wind & Fire",
    "Belle & Sebastian", "Brooks & Dunn", "Hall & Oates", "Above & Beyond",
    "Cardi B & Megan Thee Stallion", "Mumford & Sons", "Kool & The Gang",
    "Sly & The Family Stone", "Blood, Sweat & Tears", "Emerson, Lake & Palmer",
    "Crosby, Stills, Nash & Young", "Huey Lewis & The News", "KC & The Sunshine Band",
    "Martha Reeves & The Vandellas", "Diana Ross & The Supremes", "Gladys Knight & The Pips",
    "Smokey Robinson & The Miracles", "Patti LaBelle & The Bluebelles", "Harold Melvin & The Blue Notes",
    "Frankie Lymon & The Teenagers", "Gerry & The Pacemakers", "Echo & The Bunnymen",
    "Siouxsie & The Banshees", "Katrina & The Waves", "Joan Jett & The Blackhearts",
    "Tom Petty & The Heartbreakers", "Bruce Springsteen & The E Street Band",
    "Bob Seger & The Silver Bullet Band", "Southside Johnny & The Asbury Jukes",
    "Elvis Costello & The Attractions", "Ian Dury & The Blockheads", "Nick Cave & The Bad Seeds",
    "Dimitri Vegas & Like Mike", "Axwell & Ingrosso", "Axwell & Sebastian Ingrosso",
    "Skrillex & Diplo", "Aly & Fila", "HammAli & Navai", "Miyagi & Andy Panda",
    "Artik & Asti", "Azet & Zuna", "Narcotic Sound & Christian D", "Sasha Lopez & Andrea D",
    "DJ Sava & Raluka", "DJ Project & Adela Popescu", "Liviu Hodor & Mona",
    "Allexinno & Starchild", "Ariel & Radu", "Rianna & Jevon", "K-Ci & JoJo",
    "M&O", "Seals & Crofts", "Peaches & Herb", "Ashford & Simpson", "Captain & Tennille",
    "Chas & Dave", "Dave & Ansel Collins", "Maclemore & Ryan Lewis", "Sunnery James & Ryan Marciano",
    "Klaas & Mazza", "Ike & Tina Turner", "Mary J. Blige", "Method Man & Redman",
    "DJ Jazzy Jeff & The Fresh Prince", "Kid 'N Play", "Beastie Boys",
    "Run-D.M.C.", "N.W.A", "N.E.R.D.", "B.U.G. Mafia", "C.T.C.", "P.L.F.",
    "S.I.I.I.R.A.", "Dragonu' AKA 47", "S.A.M.", "Deepside Deejays", "Zdob și Zdub", "Fly Project",
    "Deepcentral", "METRO POLICE!!!", "Killa Fonic", "Nane", "Gibran Alcocer",
    "Florence + The Machine", "Marcus & Martinus", "Jack Ü",
    "XXXTENTACION × Lil Pump", "Axwell /\\ Ingrosso", "Kev X Dice",
    "P!nk", "Panic! At The Disco", "3OH!3", "Against Me!", "Godspeed You! Black Emperor",
    "The Go! Team", "Chunk! No, Captain Chunk!", "Hadouken!", "The Aquabats!", "¡Forward, Russia!",
    "Ty Dolla $ign", "A$AP Rocky", "A$AP Ferg", "A$AP Mob", "A$AP Ant", "A$AP Nast",
    "A$AP Twelvyy", "Joey Bada$$", "Ke$ha", "Curren$y", "Ca$h Out", "Nipsey Hu$$le",
    "Too $hort", "Trinidad Jame$", "Vinny Cha$e",
    "W&W", "W/.", "Anderson .Paak", "J. Cole", "T.I.", "M.I.A.",
    "Luna Amară", "Vama Veche", "Pasărea Colibri", "Frații Grime", "Celulă de Criză",
    "La Familia", "C.A.S.A. Loco", "Radio Killer", "Beach Please!", "Sișu Tudor", "Sișu", "Puffy", "Rava", "Ian", "Azteca", "Keed", "Nosfe", "Super ED", "Abi"
})

FEAT_KEYWORDS = r"\b(?:fea?t(?:uring)?|ft)(?:\.?(?!\w))|×"
TECH_FEAT = r"\b(?:fea?t(?:uring)?|ft)(?:\.?(?!\w))"

USER_AGENT = f"Sonora/{__version__} (+https://github.com/dvmizew/Sonora)"
