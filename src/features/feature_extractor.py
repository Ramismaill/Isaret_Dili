from abc import ABC, abstractmethod
import mediapipe as mp
import numpy as np


class FeatureExtractor(ABC):
    """
    Özellik çıkarımı için soyut taban sınıf.
    Tüm çıkarıcılar bu sınıftan türetilmeli.
    """

    @abstractmethod
    def extract(self, frame) -> np.ndarray:
        """Bir video karesinden özellik vektörü çıkarır."""
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        """Çıkarılan özellik vektörünün boyutunu döndürür."""
        pass


class MediaPipeExtractor(FeatureExtractor):
    """
    MediaPipe Holistic kullanarak iskelet landmark
    koordinatlarını çıkaran sınıf.
    543 anahtar nokta → normalize edilmiş koordinat dizisi.
    """

    def __init__(self, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # 33 pose + 468 face + 21 sol el + 21 sağ el = 543 nokta
        # Her nokta x, y, z → 543 * 3 = 1629 + visibility = 1662
        self._feature_size = 1662

    def extract(self, frame) -> np.ndarray:
        import cv2
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

        return self._landmarks_to_array(results)

    def _landmarks_to_array(self, results) -> np.ndarray:
        """Tüm landmark'ları tek bir numpy dizisine dönüştürür."""

        # Pose (33 nokta × 4 = 132)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 4)

        # Yüz (468 nokta × 3 = 1404)
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z]
                             for lm in results.face_landmarks.landmark]).flatten()
        else:
            face = np.zeros(468 * 3)

        # Sol el (21 nokta × 3 = 63)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z]
                                  for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand = np.zeros(21 * 3)

        # Sağ el (21 nokta × 3 = 63)
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z]
                                   for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            right_hand = np.zeros(21 * 3)

        return np.concatenate([pose, face, left_hand, right_hand])

    def get_feature_size(self) -> int:
        return self._feature_size

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Referans tabanlı normalizasyon.
        Omuzlar arası mesafeye göre ölçeklendirir.
        """
        if np.all(landmarks == 0):
            return landmarks

        # Pose landmarks'tan omuz noktaları (11 ve 12. nokta)
        left_shoulder = landmarks[11 * 4: 11 * 4 + 3]
        right_shoulder = landmarks[12 * 4: 12 * 4 + 3]
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_dist < 1e-6:
            return landmarks

        centroid = (left_shoulder + right_shoulder) / 2
        normalized = landmarks.copy()
        normalized[:132] = (landmarks[:132] - centroid[0]) / shoulder_dist

        return normalized

    def close(self):
        self.holistic.close()