"""
User-donated validation data and chat transcripts.
"""

import json
import logging
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import select, delete

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    ValidationData,
    ChatDonation,
)

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.session import SessionMixin

logger = logging.getLogger(__name__)


class DonationsMixin(SessionMixin):
    """User-donated validation data and chat transcripts."""

    def donate_validation_data(self, paper_priorities: Dict[str, Dict[str, Any]]) -> int:
        """
        Store donated paper rating data for validation purposes.

        This method accepts anonymized paper ratings from users and stores them
        in the validation_data table for improving the service.

        Parameters
        ----------
        paper_priorities : Dict[str, Dict[str, Any]]
            Dictionary mapping paper UIDs to priority data.
            Each priority data dict must contain:
            - priority (int): Rating value
            - searchTerm (str, optional): Search term associated with the rating

        Returns
        -------
        int
            Number of papers successfully donated

        Raises
        ------
        ValueError
            If paper_priorities is empty or contains invalid data format
        DatabaseError
            If database operation fails

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     priorities = {
        ...         "abc123": {"priority": 5, "searchTerm": "machine learning"},
        ...         "def456": {"priority": 4, "searchTerm": "deep learning"}
        ...     }
        ...     count = db.donate_validation_data(priorities)
        ...     print(f"Donated {count} papers")
        """
        if not paper_priorities:
            raise ValueError("No data provided")

        try:
            donated_count = 0
            for paper_uid, priority_data in paper_priorities.items():
                # Validate data format
                if not isinstance(priority_data, dict):
                    raise ValueError("Invalid data format. Expected dict with priority and searchTerm")

                priority = priority_data.get("priority", 0)
                search_term = priority_data.get("searchTerm", None)

                # Create validation data entry
                validation_entry = ValidationData(paper_uid=paper_uid, priority=priority, search_term=search_term)
                self.session.add(validation_entry)
                donated_count += 1

            # Commit all changes
            self.session.commit()

            logger.info(f"Successfully donated {donated_count} papers to validation data")
            return donated_count

        except ValueError:
            # Re-raise validation errors without rollback
            raise
        except Exception as e:
            self._rollback()
            logger.error(f"Error donating validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to donate validation data: {e}")

    def donate_chat_transcript(self, rating: str, transcript: List[Dict[str, str]]) -> int:
        """
        Store a donated chat transcript with thumbs up/down feedback.

        This method accepts an anonymized chat transcript and a rating from
        the user and stores it for improving the chat system.

        Parameters
        ----------
        rating : str
            User feedback rating, must be 'up' or 'down'.
        transcript : List[Dict[str, str]]
            List of message dicts, each with 'role' and 'text' keys.

        Returns
        -------
        int
            ID of the stored donation entry.

        Raises
        ------
        ValueError
            If rating is invalid or transcript is empty/malformed.
        DatabaseError
            If database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     transcript = [
        ...         {"role": "user", "text": "What papers discuss transformers?"},
        ...         {"role": "assistant", "text": "Here are some relevant papers..."}
        ...     ]
        ...     donation_id = db.donate_chat_transcript("up", transcript)
        """
        if rating not in ("up", "down"):
            raise ValueError(f"Invalid rating: {rating}. Must be 'up' or 'down'.")

        if not transcript or not isinstance(transcript, list):
            raise ValueError("Transcript must be a non-empty list of messages.")

        for msg in transcript:
            if not isinstance(msg, dict) or "role" not in msg or "text" not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'text' keys.")

        try:
            donation = ChatDonation(
                rating=rating,
                transcript=json.dumps(transcript),
            )
            self.session.add(donation)
            self.session.commit()

            logger.info(f"Successfully donated chat transcript (rating={rating})")
            return cast(int, donation.id)

        except ValueError:
            raise
        except Exception as e:
            self._rollback()
            logger.error(f"Error donating chat transcript: {e}", exc_info=True)
            raise DatabaseError(f"Failed to donate chat transcript: {e}")

    def get_chat_donations(
        self,
        limit: Optional[int] = None,
        rating: Optional[str] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve donated chat transcripts from the database.

        Parameters
        ----------
        limit : int, optional
            Maximum number of entries to return. Returns all if None.
        rating : str, optional
            Filter by rating ('up' or 'down'). Returns all ratings if None.
        offset : int, optional
            Number of entries to skip for pagination (default: 0).

        Returns
        -------
        list of dict
            List of donation dicts, each containing:
            - id (int): Entry ID.
            - rating (str): 'up' or 'down'.
            - transcript (list): Parsed list of message dicts.
            - donated_at (datetime): Donation timestamp.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     donations = db.get_chat_donations(limit=10, rating='up')
        """
        try:
            stmt = select(ChatDonation)
            if rating is not None:
                stmt = stmt.where(ChatDonation.rating == rating)
            stmt = stmt.order_by(ChatDonation.donated_at.desc())
            if offset:
                stmt = stmt.offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)

            results = []
            for entry in self.session.execute(stmt).scalars().all():
                try:
                    transcript = json.loads(entry.transcript)
                except Exception:
                    transcript = []
                results.append(
                    {
                        "id": entry.id,
                        "rating": entry.rating,
                        "transcript": transcript,
                        "donated_at": entry.donated_at,
                    }
                )
            return results

        except Exception as e:
            logger.error(f"Error retrieving chat donations: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve chat donations: {e}")

    def get_chat_donation_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for donated chat transcripts.

        Returns
        -------
        dict
            Statistics dict containing:
            - total (int): Total number of donations.
            - up (int): Number of thumbs-up donations.
            - down (int): Number of thumbs-down donations.
            - avg_turns (float): Average number of turns per transcript.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     stats = db.get_chat_donation_stats()
        ...     print(f"Total donations: {stats['total']}")
        """
        try:
            all_entries = self.session.execute(select(ChatDonation)).scalars().all()
            total = len(all_entries)
            up = sum(1 for e in all_entries if e.rating == "up")
            down = sum(1 for e in all_entries if e.rating == "down")
            turn_counts = []
            for entry in all_entries:
                try:
                    turns = len(json.loads(str(entry.transcript)))
                except Exception:
                    turns = 0
                turn_counts.append(turns)
            avg_turns = sum(turn_counts) / total if total > 0 else 0.0
            return {"total": total, "up": up, "down": down, "avg_turns": avg_turns}

        except Exception as e:
            logger.error(f"Error retrieving chat donation stats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve chat donation stats: {e}")

    def get_validation_data(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve donated validation (interesting-paper) data from the database.

        Parameters
        ----------
        limit : int, optional
            Maximum number of entries to return. Returns all if None.
        offset : int, optional
            Number of entries to skip for pagination (default: 0).

        Returns
        -------
        list of dict
            List of validation data dicts, each containing:
            - id (int): Entry ID.
            - paper_uid (str): Paper UID.
            - priority (int): Priority rating.
            - search_term (str or None): Associated search term.
            - donated_at (datetime): Donation timestamp.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     data = db.get_validation_data(limit=20)
        """
        try:
            stmt = select(ValidationData).order_by(ValidationData.donated_at.desc())
            if offset:
                stmt = stmt.offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)

            return [
                {
                    "id": entry.id,
                    "paper_uid": entry.paper_uid,
                    "priority": entry.priority,
                    "search_term": entry.search_term,
                    "donated_at": entry.donated_at,
                }
                for entry in self.session.execute(stmt).scalars().all()
            ]

        except Exception as e:
            logger.error(f"Error retrieving validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve validation data: {e}")

    def get_validation_data_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for donated validation (interesting-paper) data.

        Returns
        -------
        dict
            Statistics dict containing:
            - total (int): Total number of donated paper ratings.
            - unique_papers (int): Number of distinct paper UIDs.
            - avg_priority (float): Average priority rating.
            - priority_distribution (dict): Count per priority value.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     stats = db.get_validation_data_stats()
        ...     print(f"Total data donations: {stats['total']}")
        """
        try:
            all_entries = self.session.execute(select(ValidationData)).scalars().all()
            total = len(all_entries)
            unique_papers = len({e.paper_uid for e in all_entries})
            priorities = [int(e.priority) for e in all_entries]
            avg_priority = sum(priorities) / total if total > 0 else 0.0
            distribution: Dict[int, int] = {}
            for p in priorities:
                distribution[p] = distribution.get(p, 0) + 1
            return {
                "total": total,
                "unique_papers": unique_papers,
                "avg_priority": avg_priority,
                "priority_distribution": distribution,
            }

        except Exception as e:
            logger.error(f"Error retrieving validation data stats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve validation data stats: {e}")

    def delete_chat_donations(self, ids: Optional[List[int]] = None) -> int:
        """
        Delete donated chat transcripts.

        Parameters
        ----------
        ids : list of int, optional
            List of donation IDs to delete. If None, deletes all donations.

        Returns
        -------
        int
            Number of donations deleted.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     deleted = db.delete_chat_donations()
        ...     print(f"Deleted {deleted} donations")
        """
        try:
            stmt = delete(ChatDonation)
            if ids is not None:
                stmt = stmt.where(ChatDonation.id.in_(ids))
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            logger.info(f"Deleted {count} chat donation(s)")
            return count

        except Exception as e:
            self._rollback()
            logger.error(f"Error deleting chat donations: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete chat donations: {e}")

    def delete_validation_data(self, ids: Optional[List[int]] = None) -> int:
        """
        Delete donated validation (interesting-paper) data.

        Parameters
        ----------
        ids : list of int, optional
            List of entry IDs to delete. If None, deletes all validation data.

        Returns
        -------
        int
            Number of entries deleted.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     deleted = db.delete_validation_data()
        ...     print(f"Deleted {deleted} entries")
        """
        try:
            stmt = delete(ValidationData)
            if ids is not None:
                stmt = stmt.where(ValidationData.id.in_(ids))
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            logger.info(f"Deleted {count} validation data entry(ies)")
            return count

        except Exception as e:
            self._rollback()
            logger.error(f"Error deleting validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete validation data: {e}")
