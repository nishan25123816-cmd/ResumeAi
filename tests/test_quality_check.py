import unittest

from src.quality_check import looks_like_low_quality_ocr


class QualityCheckTests(unittest.TestCase):
    def test_rejects_garbled_ocr_text(self):
        bad_text = '''
        S02U0UOdXO J0SN [nyBurueaw epiaosd yey) Awapn - sewawepuny QHD BUD + sSyonpoud jeyBip eyees> pue ‘511145 Jexuyoa; Aw dojaaap Ajsnonunuoo “Swans B6ay09 u! so2ueWUO}Iad SUP 205 PepAMY e29suNOD - sxseg UBISEQ XIN + peload enqenceas oy eincgnucs ~swesGoud jesmyn> pur swans adyynw UI PABRUN}OA dweJepo eau
        '''
        self.assertTrue(looks_like_low_quality_ocr(bad_text))

    def test_accepts_normal_resume_text(self):
        good_text = '''
        John Smith
        Senior Software Engineer
        john.smith@email.com | +1 (415) 555-1234

        Experience
        Built scalable backend services using Python, Java, and AWS.
        Led a team of 5 engineers and improved deployment reliability by 30%.

        Skills
        Python, SQL, AWS, Docker, Kubernetes, Leadership
        '''
        self.assertFalse(looks_like_low_quality_ocr(good_text))


if __name__ == "__main__":
    unittest.main()
